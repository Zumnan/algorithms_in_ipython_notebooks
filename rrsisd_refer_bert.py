import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import random
import transformers
from transformers import BertTokenizer
from refer.refer import REFER

from args import get_parser

import re
import nltk
from nltk.tokenize import word_tokenize


import os
import sys

# Set the NLTK data path to offline directory
nltk_data_path = "/10T/students/doctor/2025/zum/anaconda3/share/nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
os.environ['NLTK_DATA'] = nltk_data_path

# Add the directory containing the modules to sys.path
module_dir = os.path.join(os.path.dirname(__file__), 'refer')  # Adjust 'lib' as needed
if module_dir not in sys.path:
    sys.path.append(module_dir)

# Import local modules with error handling
# Try to import mmseg utils, but provide fallback if not available
try:
    from refer.refer import REFER
except ImportError:
    print("Warning: Could not import REFER from refer.refer")
    # Create a simple fallback logger
    import logging
    def get_root_logger(log_file=None, log_level=logging.INFO):
        logger = logging.getLogger()
        if not logger.handlers:
            logging.basicConfig(
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=log_level)
        return logger

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


def add_random_boxes(img, min_num=20, max_num=60, size=32):
    h,w = size, size
    img = np.asarray(img).copy()
    img_size = img.shape[1]
    boxes = []
    num = random.randint(min_num, max_num)
    for k in range(num):
        y, x = random.randint(0, img_size-w), random.randint(0, img_size-h)
        img[y:y+h, x: x+w] = 0
        boxes.append((x,y,h,w))
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


def _find_sublist_indices(tokens, sub_tokens):
    if not sub_tokens or not tokens:
        return []
    matches = []
    max_start = len(tokens) - len(sub_tokens)
    for i in range(max_start + 1):
        if tokens[i:i + len(sub_tokens)] == sub_tokens:
            matches.append(i)
    return matches


def _load_tokenizer(args):
    cache_dir = getattr(args, "hf_cache_dir", None) or None
    language_encoder = getattr(args, "language_encoder", "bert")
    if language_encoder == "clip":
        clip_path = getattr(args, "clip_model_path", "")
        if not clip_path or not os.path.isdir(clip_path):
            raise RuntimeError(f"CLIP tokenizer path not found: {clip_path}")
        clip_tok_cls = getattr(transformers, "CLIPTokenizerFast", None) or getattr(transformers, "CLIPTokenizer", None)
        if clip_tok_cls is not None:
            return clip_tok_cls.from_pretrained(clip_path, cache_dir=cache_dir, local_files_only=True)
        auto_cls = getattr(transformers, "AutoTokenizer", None)
        if auto_cls is not None:
            return auto_cls.from_pretrained(clip_path, cache_dir=cache_dir, local_files_only=True)
        gpt_cls = getattr(transformers, "GPT2TokenizerFast", None) or getattr(transformers, "GPT2Tokenizer", None)
        if gpt_cls is not None:
            vocab_file = os.path.join(clip_path, "vocab.json")
            merges_file = os.path.join(clip_path, "merges.txt")
            tok = gpt_cls(vocab_file=vocab_file, merges_file=merges_file)
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
            return tok
        raise RuntimeError("CLIP tokenizer classes not available in this transformers build.")
    bert_path = getattr(args, "bert_tokenizer", "bert-base-uncased")
    local_only = os.path.isdir(bert_path)
    return BertTokenizer.from_pretrained(bert_path, cache_dir=cache_dir, local_files_only=local_only)


def _safe_word_tokenize(text: str):
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _safe_pos_tag(tokens):
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        return [(tok, "NN") for tok in tokens]


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = int(getattr(args, "max_seq_length", 22))

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        num_images_to_mask = int(len(ref_ids) * 0.2)
        self.images_to_mask = random.sample(ref_ids, num_images_to_mask)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = _load_tokenizer(args)

        # for ground target and spatial position
        self.target_masks = []
        self.position_masks = []

        self.sentense_raw = []
        self.pp_phrase = []

        # debug
        self.max_len = 0

        # for RRSIS-D dataset
        self.target_cls = {"airplane", "airport", "golf field", "expressway service area", "baseball field","stadium",
                      "ground track field", "storage tank", "basketball court", "chimney", "tennis court", "overpass",
                      "train station", "ship", "expressway toll station", "dam", "harbor", "bridge", "vehicle",
                      "windmill"}

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            target_for_ref = []
            position_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                target_masks = [0] * self.max_tokens
                position_masks = [0] * self.max_tokens

                encoding = self.tokenizer.encode_plus(
                    text=sentence_raw,
                    add_special_tokens=True,
                    max_length=self.max_tokens,
                    truncation=True,
                )
                input_ids = encoding["input_ids"]
                input_len = len(input_ids)

                padded_input_ids[:input_len] = input_ids
                attention_mask[:input_len] = [1] * input_len

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

                # extract the ground object (align with BERT tokens)
                self.sentense_raw.append(sentence_raw)
                if input_len >= 2:
                    token_ids = input_ids[1:-1]
                else:
                    token_ids = input_ids[1:]
                bert_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                for cls in self.target_cls:
                    tokenized_cls = self.tokenizer.tokenize(cls)
                    for match_start in _find_sublist_indices(bert_tokens, tokenized_cls):
                        start = match_start + 1  # [CLS]
                        end = min(start + len(tokenized_cls), self.max_tokens - 1)
                        target_masks[start:end] = [1] * (end - start)

                target_for_ref.append(torch.tensor(target_masks).unsqueeze(0))

                # extract the spatial position
                grammar = r"""
                PP: {<IN><DT>?<JJ.*>?<NN>}
                    {<IN><DT>?<JJ.*>?<JJ>}
                    {<IN><DT>?<JJ.*><VBD>}
                """
                chunkr = nltk.RegexpParser(grammar)
                tokenized_sentence = _safe_word_tokenize(sentence_raw)
                # grammar parsing
                tree = chunkr.parse(_safe_pos_tag(tokenized_sentence))
                pp_phrases = []
                for subtree in tree.subtrees():
                    if subtree.label() == 'PP':
                        pp_phrases.append(' '.join(word for word, pos in subtree.leaves()))

                new_pp_phrase = []
                for phrase in pp_phrases:
                    if not re.findall("of", phrase):
                        new_pp_phrase.append(phrase)

                if len(new_pp_phrase) > 0:
                    for pp in new_pp_phrase:
                        tokenized_pos = self.tokenizer.tokenize(pp)
                        for match_start in _find_sublist_indices(bert_tokens, tokenized_pos):
                            start = match_start + 1  # [CLS]
                            end = min(start + len(tokenized_pos), self.max_tokens - 1)
                            position_masks[start:end] = [1] * (end - start)

                self.pp_phrase.append(new_pp_phrase)
                position_for_ref.append(torch.tensor(position_masks).unsqueeze(0))
                # if there are no pp for this sentence, fall back to attention mask
                if torch.sum(position_for_ref[-1]) == 0:
                    position_for_ref[-1] = attentions_for_ref[-1]

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
            self.target_masks.append(target_for_ref)
            self.position_masks.append(position_for_ref)


    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name']))
        if self.split == 'train' and this_ref_id in self.images_to_mask:
            img = add_random_boxes(img)

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        sentence = ref[0]['sentences'][0]['raw']
        save_prefix = str(ref[0]['image_id']) + "_" + sentence

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            # Leisen Debug: write the input images and labels
            SHOW_INPUT = False
            if SHOW_INPUT:
                import cv2
                save_dir = "experiments/input_vis"

                # write in the type of image and label
                img.save(os.path.join(save_dir, save_prefix + "_image.png"))
                mask = ref_mask * 255
                cv2.imwrite(os.path.join(save_dir, save_prefix + "_label.png"), mask)

            img, target = self.image_transforms(img, annot)

        if self.eval_mode:
            choice_sent = 0
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]
        target_mask = self.target_masks[index][choice_sent]
        position_mask = self.position_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask, target_mask, position_mask, save_prefix
