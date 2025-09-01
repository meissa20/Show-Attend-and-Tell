import argparse, json
from collections import Counter
import random

def generate_json_data(data_path, max_captions_per_image, min_word_count):
    word_count = Counter()
    train_img_paths = []
    train_caption_tokens = []
    validation_img_paths = []
    validation_caption_tokens = []
    
    with open(f"{data_path}/captions/captions.txt", "r") as f:
        lines = f.readlines()
        random.shuffle(lines)
        split_idx = int(0.9 * len(lines))
        train_lines = lines[:split_idx]
        val_lines = lines[split_idx:]

        for line in train_lines:
            filename, caption = line.strip().split(",", 1)
            tokens = caption.split()
            train_img_paths.append(f"/kaggle/input/image-captioning-dataset/Images/{filename}")
            train_caption_tokens.append(tokens)
            word_count.update(tokens)   # The model won’t see rare words enough during training to learn good embeddings.
                                        # Keeping them makes the vocabulary huge and training inefficient.
                                        # as their embeddings are poorly trained (close to random noise).

        for line in val_lines:
            filename, caption = line.strip().split(",", 1)
            tokens = caption.split()
            validation_img_paths.append(f"/kaggle/input/image-captioning-dataset/Images/{filename}")
            validation_caption_tokens.append(tokens)
            word_count.update(tokens)

    words = [word for word in word_count.keys() if word_count[word] >= min_word_count]   # Take all words from captions, but only keep the frequent ones
    word_dict = {word: idx + 4 for idx, word in enumerate(words)}  # Make a dictionary mapping words 
    word_dict['<start>'] = 0                                       # (starting at 4, as 0–3 are reserved for special tokens).
    word_dict['<eos>'] = 1
    word_dict['<unk>'] = 2
    word_dict['<pad>'] = 3    
    
    with open(data_path + '/word_dict.json', 'w') as f:
        json.dump(word_dict, f)
        
    train_captions = process_caption_tokens(train_caption_tokens, word_dict, max_captions_per_image)
    validation_captions = process_caption_tokens(validation_caption_tokens, word_dict, max_captions_per_image)

    with open(data_path + '/train_img_paths.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open(data_path + '/val_img_paths.json', 'w') as f:
        json.dump(validation_img_paths, f)
    with open(data_path + '/train_captions.json', 'w') as f:
        json.dump(train_captions, f)
    with open(data_path + '/val_captions.json', 'w') as f:
        json.dump(validation_captions, f)


def process_caption_tokens(caption_tokens, word_dict, max_length):
    captions = []
    for tokens in caption_tokens:
        token_idxs = [word_dict[token] if token in word_dict else word_dict['<unk>'] for token in tokens]
        captions.append(
            [word_dict['<start>']] + token_idxs + [word_dict['<eos>']] +
            [word_dict['<pad>']] * (max_length - len(tokens)))

    return captions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate json files')
    parser.add_argument('--data-path', type=str, default='../data')
    parser.add_argument('--max-captions', type=int, default=40,
                        help='maximum number of captions per image')
    parser.add_argument('--min-word-count', type=int, default=2,
                        help='minimum number of occurences of a word to be included in word dictionary')
    args = parser.parse_args()

    generate_json_data(args.data_path, args.max_captions, args.min_word_count)
