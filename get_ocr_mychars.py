import os
import shutil
from tqdm import tqdm

def get_chars_to_txt(txt_file, save_file, split=' ', save_split='\n'):
    '''
    从txt中获取所有单个字符集,要求txt格式为图片[TAB或者空格]字符串
    :param save_file:
    :param split: 图片和字符串分割符
    :return:
    '''
    chars_list = []
    if isinstance(txt_file, list):  # txt list files
        for file in txt_file:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.read().rstrip('\n').split('\n')
                for line in tqdm(lines):
                    index = line.find(split)
                    if index > 0:
                        # image_file = line[:index]
                        chars = line[index + 1:]
                        # print(image_file)
                        # print(labels)
                        for char in chars:
                            if not char in chars_list:
                                chars_list.append(char)
    else:  # just a txt file
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.read().rstrip('\n').split('\n')
            for line in tqdm(lines):
                index = line.find(split)
                if index > 0:
                    # image_file = line[:index]
                    chars = line[index + 1:]
                    # print(image_file)
                    # print(labels)
                    for char in chars:
                        if not char in chars_list:
                            chars_list.append(char)
    with open(save_file, 'w') as f:
        f.write(save_split.join(sorted(chars_list)))


if __name__ == '__main__':
    txt_file = ['data/labels/train.txt', 'data/labels/val.txt']
    save_file = 'data/mychars.txt'
    get_chars_to_txt(txt_file, save_file, split='\t', save_split='')
