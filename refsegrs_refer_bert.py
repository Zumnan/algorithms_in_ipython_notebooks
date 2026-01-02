import sys
import os.path as osp

import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import transformers
from transformers import BertTokenizer
from args import parse_args
import cv2

import re
import nltk
from nltk.tokenize import word_tokenize

# Dataset configuration initialization
args = parse_args()

if args.dataset == "refsegrs":
    images_subdir = "images2"
    masks_subdir = "masks2"
else:
    images_subdir = "images2"
    masks_subdir = "masks2"

# Fallback to dataset-specific roots if refer_data_root wasn't injected by the caller
DATA_ROOT = args.refer_data_root or (
    args.refsegrs_data_root if args.dataset == "refsegrs" else args.rrsisd_data_root
)
IMAGES_DIR = osp.join(DATA_ROOT, images_subdir)
MASKS_DIR = osp.join(DATA_ROOT, masks_subdir)
TRAIN_SETFILE = osp.join(DATA_ROOT, "output_phrase_train.txt")
VAL_SETFILE = osp.join(DATA_ROOT, "output_phrase_val.txt")
TEST_SETFILE = osp.join(DATA_ROOT, "output_phrase_test.txt")


def _verify_path(path, description):
    if not osp.exists(path):
        sys.exit(f"{description} not found at {path}")


def build_rsris_batches(setname):
    im_dir1 = IMAGES_DIR
    seg_label_dir = MASKS_DIR
    _verify_path(im_dir1, "Image directory")
    _verify_path(seg_label_dir, "Mask directory")
    if setname == 'train':
        setfile = TRAIN_SETFILE
    if setname == 'val':
        setfile = VAL_SETFILE
    if setname == 'test':
        setfile = TEST_SETFILE

    _verify_path(setfile, f"{setname} split file")

    train_ids = []
    nn = 0
    imgnames = set()
    imname = 'start'
    all_imgs1 = []
    all_labels = []
    all_sentences = []

    test_sentence = []

    with open(setfile, 'r') as rf:
        rlines = rf.readlines()
        for idx, line in enumerate(rlines):
            lsplit = line.split(' ')
            im_name1 = osp.join(im_dir1, lsplit[0] + '.tif')
            seg = osp.join(seg_label_dir, lsplit[0] + '.tif')
            del (lsplit[0])
            sentence = ' '.join(lsplit)
            sent = sentence

            im_1 = im_name1
            label_mask = seg
            all_imgs1.append(im_name1)
            all_labels.append(label_mask)
            all_sentences.append(sent)

    print("Dataset Loaded.")
    return all_imgs1, all_labels, all_sentences


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
        self.max_tokens = 20

        all_imgs1, all_labels, all_sentences = build_rsris_batches(self.split)
        self.sentences = all_sentences
        self.imgs1 = all_imgs1
        self.labels = all_labels

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = _load_tokenizer(args)

        self.target_masks = []
        self.position_masks = []

        self.sentences_raw = []
        self.pp_phrase = []

        # debug
        self.max_len = 0

        # for RefSegRS dataset
        self.target_cls = {"road", "vehicle", "car", "van", "building", "truck", "trailer", "bus",
                      "road marking", "bikeway", "sidewalk", "tree", "low vegetation",
                      "impervious surface"}

        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in range(len(self.imgs1)):
            img_sentences = [self.sentences[r]]
            sentences_for_ref = []
            attentions_for_ref = []

            target_for_ref = []
            position_for_ref = []

            for i, el in enumerate(img_sentences):
                sentence_raw = el
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                target_masks = [0] * self.max_tokens
                position_masks = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

                # extract the ground object
                # print(sentence_raw)
                self.sentences_raw.append(sentence_raw)
                tokenized_sentence = word_tokenize(sentence_raw)

                for cls in self.target_cls:
                    if re.findall(cls, sentence_raw):

                        tokenized_cls = word_tokenize(cls)
                        nums_cls = len(tokenized_cls)
                        index = 0
                        for i, token in enumerate(tokenized_sentence):
                            if re.findall(tokenized_cls[0], token):
                                index = i
                                break
                        target_masks[index + 1: index + nums_cls + 1] = [1] * nums_cls

                target_for_ref.append(torch.tensor(target_masks).unsqueeze(0))

                # extract the spatial position
                grammar = r"""
                PP: {<IN><DT>?<JJ.*>?<NN>}
                    {<IN><DT>?<JJ.*>?<JJ>}
                    {<IN><DT>?<JJ.*><VBD>}
                """
                chunkr = nltk.RegexpParser(grammar)
                # grammar parsing
                tree = chunkr.parse(nltk.pos_tag(tokenized_sentence))
                pp_phrases = []
                for subtree in tree.subtrees():
                    if subtree.label() == 'PP':
                        pp_phrases.append(' '.join(word for word, pos in subtree.leaves()))

                new_pp_phrase = []
                for phrase in pp_phrases:
                    if not re.findall("of", phrase):
                        new_pp_phrase.append(phrase)

                if len(new_pp_phrase) > 0:
                    tokenized_sentence = word_tokenize(sentence_raw)
                    for pp in new_pp_phrase:
                        tokenized_pos = word_tokenize(pp)
                        nums_pos = len(tokenized_pos)
                        index = 0
                        for i, token in enumerate(tokenized_sentence):
                            if tokenized_pos[0] == token:
                                index = i
                                break
                        position_masks[index + 1: index + nums_pos +1] = [1] * nums_pos

                self.pp_phrase.append(new_pp_phrase)
                position_for_ref.append(torch.tensor(position_masks).unsqueeze(0))
                # if there are no pp, the position_for_ref equals attentions_for_ref
                if torch.sum(position_for_ref[0]) == 0:
                    position_for_ref = attentions_for_ref

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
            self.target_masks.append(target_for_ref)
            self.position_masks.append(position_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.imgs1)

    def __getitem__(self, index):
        this_img1 = self.imgs1[index]

        img1 = Image.open(this_img1).convert("RGB")
        label_mask = cv2.imread(self.labels[index],2)

        ref_mask = np.array(label_mask) > 50
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")
        save_prefix = str(index) + "_" + self.sentences_raw[index][:-1]
        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img1, target = self.image_transforms(img1, annot)

        choice_sent = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask = self.attention_masks[index][choice_sent]
        target_mask = self.target_masks[index][choice_sent]
        position_mask = self.position_masks[index][choice_sent]

        return img1, target, tensor_embeddings, attention_mask, target_mask, position_mask, save_prefix
