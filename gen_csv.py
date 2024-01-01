import csv
import os
import argparse


def read_csv(file_name):
    content_list = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            content_list.append(row)
    print(len(content_list))
    return content_list

def gen_csv(content_list, csv_name):
    data_list = []
    # Add title
    data_list.append(["text"])
    # Add contents: "<s>[INST]" + query + " [/INST]  " + gpt-3.5-turbo + " </s>"
    # questionID,questionTitle,questionText,questionLink,topic,therapistInfo,therapistURL,answerText,upvotes,views
    for content in content_list:
        data_content = "<s>[INST]"+ content[0] + " [/INST]  Anwser:" + content[1] + "</s>"
        data_list.append([data_content])
    print(len(data_list))
    with open(csv_name, 'w', newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data_list)

def clean_data(content_list):
    new_content_list = []
    for content in content_list:
        if(len(content) != 1):
            continue
        dirty_data = content[0]
        # Remove data with URL
        if ".com" in dirty_data or ".org" in dirty_data or "http" in dirty_data: # delete link
            continue
        # Remove long data
        if len(dirty_data) > 2048:
            continue
        # Remove abnormal data (When using Notepad++ encoding to UTF-8, contain "??")
        if "??" in dirty_data:
            continue
        new_content_list.append(content)
    return new_content_list

def main(origin_path:str, new_path:str):
    origin_file = origin_path
    new_file = new_path
    content_list = read_csv(origin_file)
    content_list = clean_data(content_list)
    gen_csv(content_list, new_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The IMHI benchmark.')
    parser.add_argument('--origin_path', type=str)
    parser.add_argument('--new_path', type=str)
    args = parser.parse_args()
    args = vars(args)
    main(**args)
    
        