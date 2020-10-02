#-*-coding:utf-8 -*-
'''
@File    :   kdxf.py
@Time    :   2020-09-30 17:28:46
@Author  :   GuoLiuFang
@Version :   0.1
@Contact :   guoliufang001@ke.com
@License :   (C)Copyright 2018-2020, KeAI, KeOCR
@Desc    :   用于科大讯飞数据清洗。使用python3 run解决很多问题
/usr/local/bin/python3.7
/usr/local/bin/pip3.7
'''
import json
import soundfile as sf
import codecs
import argparse
import glob
import os
import re
from tqdm import tqdm
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input_txt_dir",
    default="/KeAI/STT/speech_recognition/data/data_phone/urls",
    type=str,
    # 一共17个csv
    help="Directory where to read the phone data. (default: %(default)s)"
)
parser.add_argument(
    "--input_audio_dir",
    default="/KeAI/STT/speech_recognition/data/data_phone/data",
    type=str,
    # 一共17个csv
    help="Directory where to read the phone data. (default: %(default)s)"
)
parser.add_argument(
    "--output_dir",
    # 存储文件manifest和音频wav
    default="/KeAI/STT/speech_recognition/data/data_phone/data_kdxf",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
args = parser.parse_args()
def clean_line(line):
    # 去掉标点符号，和进行空格删除。
    tmp = re.sub(r'[^\w\s]', '', line)
    return ''.join(i.strip() for i in tmp.split())

def prepare_dataset(input_txt_dir, input_audio_dir, output_dir, manifest_prefix):
    # 控制合并条件，目前为了保证最大的兼容性，强制为0.
    combine_threshold = 0
    for csv_url in tqdm(glob.glob(input_txt_dir + "/*.csv")):
        print('*'*50)
        print("current is deal with ", csv_url)
        print('*'*50)
        # 按照城市分开存储
        json_lines_with_digit_line = []
        json_lines_no_digit_line = []
        json_lines_with_digit = []
        json_lines_no_digit = []
        # if '_head' in csv_url:
        #     audio_file_dir = os.path.basename(csv_url.replace('_head', "")[:-4]) + "_url"
        audio_file_dir = os.path.basename(csv_url[:-4]) + "_url"
        audio_segment = os.path.join(output_dir, "wav", audio_file_dir)
        if not os.path.exists(audio_segment):
            os.makedirs(audio_segment)
        with codecs.open(csv_url, 'r', encoding='gbk') as f:
            for index, line in enumerate(f.readlines()):
                if index != 0:
                    url = line.split(',')[0]
                    # 通过url中的文件地址，把，拿到语音文件
                    file_name = url.split("/")[-1]
                    audio_filepath = os.path.join(input_audio_dir, audio_file_dir, file_name)
                    print(audio_filepath)
                    content = line[len(url)+1:]
                    asr_content = json.loads(content)
                    # combine_threshold进行内容和语音的合并
                    # 预置一个元素。（[time_list], [content_list]）
                    service_list = [([-1], "")]
                    customer_list  = [([-1], "")]
                    for inx, line in enumerate(asr_content):
                        qu_jian = line['time']
                        qu_jian_list = [int(i) for t in qu_jian.split() for i in t.split(',')]
                        curr_content = line['content']
                        if line["channel"] == "坐席:":
                            # TODO: 代码复用的优化。
                            # 如果存量的最后一个，跟增量的第一个相等的话，就对先前的那个元素进行更新。
                            pre_max = service_list[-1][0][-1]
                            current_min = qu_jian_list[0]
                            # 目前使用【相等】
                            if current_min - pre_max == combine_threshold:
                                last_list = service_list[-1][0]
                                last_content = service_list[-1][1]
                                new_last_list = last_list + qu_jian_list
                                del service_list[-1]
                                service_list.append((new_last_list, last_content + " " + curr_content))
                            else:
                                service_list.append((qu_jian_list, curr_content))
                        if line["channel"] == "客户:":
                            # TODO: 代码复用的优化。
                            # 如果存量的最后一个，跟增量的第一个相等的话，就对先前的那个元素进行更新。
                            pre_max = service_list[-1][0][-1]
                            current_min = qu_jian_list[0]
                            # 目前使用【相等】
                            if current_min - pre_max == combine_threshold:
                                last_list = customer_list[-1][0]
                                last_content = customer_list[-1][1]
                                new_last_list = last_list + qu_jian_list
                                del customer_list[-1]
                                customer_list.append((new_last_list, last_content + " " + curr_content))
                            else:
                                customer_list.append((qu_jian_list, curr_content))
                    del service_list[0]
                    del customer_list[0]
                    record_list = service_list + customer_list
                    service_len = len(service_list)
                    # combine 语音的合并完成。接下来进行切割和语料处理。
                    # 这里的思想是进行数据分层。不需要做那么多的处理。
                    for inx, record in enumerate(record_list):
                        content = record[1]
                        qu_jian = record[0]
                        # start 和 stop全部是frames。。
                        start, stop = qu_jian[0], qu_jian[-1]
                        # 【* 8】是依据科大讯飞的数据观察出来的。
                        if os.path.exists(audio_filepath):
                            audio_data, audio_samplerate = sf.read(audio_filepath, frames=(stop - start) * 8, start=start * 8)
                        else:
                            print("*" * 500)
                            print(f"---{audio_filepath}---is not exsits")
                            print("*" * 500)
                            break
                        # TODO：单声道和多声道的问题。
                        # inx < service_len就是坐席。
                        if inx < service_len:
                            wav_path = os.path.join(audio_segment, file_name[:-4]) + "_service_" + str(inx) + ".wav"
                            sf.write(wav_path, audio_data[:, 0], samplerate=audio_samplerate , subtype='PCM_16')
                        else:
                            wav_path = os.path.join(audio_segment, file_name[:-4]) + "_customer_" + str(inx) + ".wav"
                            sf.write(wav_path, audio_data[:, 1], samplerate=audio_samplerate , subtype='PCM_16')
                        # 这里增加核心的逻辑，因为是写文件，可以写多个。
                        # 以是否含有数字【0-9】，作为分割条件，存储为两个文件，digit和no_digit
                        if any(i.isdigit() for i in content):
                            json_lines_with_digit.append(
                                json.dumps(
                                    {
                                        'audio_filepath': wav_path,
                                        'duration': float(len(audio_data)) / audio_samplerate,
                                        # 'duration_ss': float(stop - start) * 8 / audio_samplerate,
                                        # 写原始和经过处理的
                                        'text': content
                                    },
                                    ensure_ascii=False))
                            json_lines_with_digit_line.append(
                                json.dumps(
                                    {
                                        'audio_filepath': wav_path,
                                        'duration': float(len(audio_data)) / audio_samplerate,
                                        # 'duration_ss': float(stop - start) * 8 / audio_samplerate,
                                        # 写原始和经过处理的
                                        'text': clean_line(content)
                                    },
                                    ensure_ascii=False))
                        else:
                            json_lines_no_digit.append(
                                json.dumps(
                                    {
                                        'audio_filepath': wav_path,
                                        'duration': float(len(audio_data)) / audio_samplerate,
                                        # 'duration_ss': float(stop - start) * 8 / audio_samplerate,
                                        'text': content
                                    },
                                    ensure_ascii=False))
                            json_lines_no_digit_line.append(
                                json.dumps(
                                    {
                                        'audio_filepath': wav_path,
                                        'duration': float(len(audio_data)) / audio_samplerate,
                                        # 'duration_ss': float(stop - start) * 8 / audio_samplerate,
                                        'text': clean_line(content)
                                    },
                                    ensure_ascii=False))

        if not os.path.exists(os.path.join(output_dir, "transcript", audio_file_dir)):
            os.makedirs(os.path.join(output_dir, "transcript", audio_file_dir))
        with codecs.open(os.path.join(output_dir, "transcript", audio_file_dir, manifest_prefix + "_no_digit_line") , 'w', 'utf-8') as fout:
            for line in json_lines_no_digit_line:
                fout.write(line + '\n')
        with codecs.open(os.path.join(output_dir, "transcript", audio_file_dir, manifest_prefix + "_no_digit") , 'w', 'utf-8') as fout:
            for line in json_lines_no_digit:
                fout.write(line + '\n')
        with codecs.open(os.path.join(output_dir, "transcript", audio_file_dir, manifest_prefix + "_with_digit_line") , 'w', 'utf-8') as fout:
            for line in json_lines_with_digit_line:
                fout.write(line + '\n')
        with codecs.open(os.path.join(output_dir, "transcript", audio_file_dir, manifest_prefix + "_with_digit") , 'w', 'utf-8') as fout:
            for line in json_lines_with_digit:
                fout.write(line + '\n')

def main():
    prepare_dataset(
        input_txt_dir=args.input_txt_dir,
        # input_txt_dir="/KeAI/STT/speech_recognition/data/data_phone/urls/url_phone_beijing_2020-03_2020-05_head.csv",
        input_audio_dir=args.input_audio_dir,
        output_dir=args.output_dir,
        manifest_prefix=args.manifest_prefix
    )

if __name__ == '__main__':
    main()
