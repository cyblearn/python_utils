#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021.02.04

@author: chenyingbo
"""

# ------------------------------------------------------------------  intfs  ----------------------------------------------------------------------- #
def getFiles(dir, suff='')                                                                                              # 获取文件列表
def files_find(src_dir, suff_pos=[], suff_neg=[])                                                                       # 从文件夹获取符合要求的文件列表
def files_copy(src_dir, dst_dir, suff_pos=[], suff_neg=[], need_idx=False)                                              # 文件重命名，考虑到重复
def files_del(dir, pos_str='_____', neg_str='______')                                                                   # 批量删除文件
def augment_add_noise(source_dir, noise_dir, source_suff='_IAR_',dst_dir='', noise_suff='nsout', SR=16000, times=1)     # 数据加噪声
def micarray_intf(dir)                                                                                                  # 一个处理多通道文件的示例
def split_audio(audio_path, dst_dir, dur=2, name_pref='', SR=16000, padding=False)                                      # 分割音频
def txt2wav(txt_name, wav_name)                                                                                         # txt转wav
# ------------------------------------------------------------------  funcs  ----------------------------------------------------------------------- #


# 递归获取文件列表，可选后缀。返回完整路径列表。如果suff为空则返回全部文件。
# 由于'1234'.find('')=0，所以getFiles不传入参数时默认返回所有文件。
def getFiles(dir, suff=''):
    flist = []
    def getFlist(path, p2_suff):
        lsdir = os.listdir(path)
        dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
        if dirs:
            for i in dirs:
                getFlist(os.path.join(path, i), p2_suff)
        files = [i for i in lsdir if os.path.isfile(os.path.join(path, i))]
        for file in files:
            if file.find(p2_suff) >= 0:
                file_path = os.path.join(path, file)
                flist.append(file_path)
        return flist
    getFlist(dir, suff)
    return flist

# 寻找文件夹中的文件，返回文件列表，满足列表中的所有条件。pos列表中是且的关系，neg是或的关系
# 文件名需包含suff_pos所有，且不包含suff_neg中任意一个
def files_find(src_dir, suff_pos=[], suff_neg=[]):    
    # 先获取全部文件
    all_files_tmp = getFiles(src_dir)
    all_files = []
    
    if suff_pos == [] and suff_neg == []:
        return all_files_tmp
    
    for i in range(len(all_files_tmp)):
        is_desire = 1
        for j in range(len(suff_pos)):
            if all_files_tmp[i].find(suff_pos[j]) < 0:
                is_desire = 0
                break
        if is_desire == 1:
            
            for j in range(len(suff_neg)):
                if all_files_tmp[i].find(suff_neg[j]) >= 0:
                    is_desire = 0
                    break
        if is_desire == 1:
            all_files.append(all_files_tmp[i])
            
    return all_files
        
    

"""
功能：复制符合某种命名的文件到新的文件夹，考虑到重命名
示例：
src_dir = '/home1/train_data/asr_alexa/'
dst_dir = '/home1/train_data/asr_alexa/test1/'
files_copy(src_dir, dst_dir, suff_pos=['_PC', '_out'], suff_neg=['_IAR_'], need_idx=True)
"""
def files_copy(src_dir, dst_dir, suff_pos=[], suff_neg=[], need_idx=False):
    all_files = files_find(src_dir, suff_pos, suff_neg)
    
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    for i in range(len(all_files)):
        _, filename = os.path.split(all_files[i]);
        if need_idx == True:
            filename = str(i) + "_" + filename
        file_out_path = os.path.join(dst_dir, filename)
        shutil.copyfile(all_files[i], file_out_path)




# 批量删除带特定名称的文件。
# 删除名字中带有pos_str且不带有neg_str的
def files_del(dir, pos_str='_____', neg_str='______'):
    file_lists = getFiles(dir)
    
    for i in range(len(file_lists)):
        if file_lists[i].find(pos_str) >= 0 and file_lists[i].find(neg_str) < 0:
            os.remove(file_lists[i])

# 音频数据增广，加噪声
# 正样本叠加噪声
"""
source_dir & source_suff: 源音频文件及过滤音频用的字符串
noise_dir  & noise_suff : 噪声文件及过滤噪声用的字符串
dst_dir: 存放增广后的文件
SR： 采样率
times: 进行几倍的扩展

说明： 噪声样本长度必须大于源音频长度，否则就跳过生成。
"""
def augment_add_noise(source_dir, noise_dir, source_suff='_IAR_',dst_dir='', noise_suff='nsout', SR=16000, times=1):
    # step 0: import
    import librosa
    import os
    import numpy as np
    import wave
    import random
    
    # step 1: 获取文件列表
    audio_path_list = getFiles(source_dir, source_suff)
    noise_path_list = getFiles(noise_dir, noise_suff)
    if len(audio_path_list) == 0 or len(noise_path_list) == 0:
        print("error!! no files")
        exit(0)
    
    # step 2: 读取所有噪声到 all_noise_wav (list)
    all_noise_wav = []
    for i in range(len(noise_path_list)):
        data_tmp, _ = librosa.load(noise_path_list[i], sr=SR, mono=True)
        all_noise_wav.append(data_tmp.copy())
        
    # step 3: 为每个样本生成标注
    for i in range(times):                              # i: 轮数编号
        for j in range(len(audio_path_list)):           # j: 正样本编号
            # step 3.1: 读语音
            x, _ = librosa.load(audio_path_list[j], sr=SR, mono=True)
        
            # step 3.2: 生成随机数：随机噪声，随机索引，随机音量
            rd_noise_idx  = random.randint(0, len(all_noise_wav) - 1)
            if all_noise_wav[rd_noise_idx].shape[0] < x.shape[0]:
                continue
            rd_idx        = random.randint(0, all_noise_wav[rd_noise_idx].shape[0] - x.shape[0] - 1)
            rd_vol        = (float)(random.randint(6, 20)) / 100
    
            # step 3.3: 叠加
            x = x * rd_vol  + all_noise_wav[rd_noise_idx][rd_idx : rd_idx + x.shape[0]]
            x = x * 32768
            
            # step 3.4: 写入文件
            (filepath,tempfilename) = os.path.split(audio_path_list[j])
            (filename,extension) = os.path.splitext(tempfilename)
            dst_name = tempfilename + "_rnd_" + str(rd_noise_idx) + "_" + str(rd_idx) + "_" + str((int)(rd_vol * 100)) + extension
            dst_path = os.path.join(dst_dir, dst_name)
            
            x = x.astype(np.short)
            f = wave.open(dst_path, "wb")
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SR)
            # 将wav_data转换为二进制数据写入文件
            f.writeframes(x.tostring())
            f.close()
 
# 找出列表里所有符合特定命名规则的多通道文件，如wav_suffs = ['_ch0.wav', '_ch1.wav', '_ch2.wav', '_ch3.wav']；
# 然后以若干个为一组进行某种处理
def micarray_intf(dir):
    wav_suffs = ['_ch0.wav', '_ch1.wav', '_ch2.wav', '_ch3.wav']
    wav_path_list_tmp = getFiles(dir, suff=".wav")
    wav_path_list = []
    
    # 先去除掉后缀不对的文件
    for i in range(len(wav_path_list_tmp)):
        suff = wav_path_list_tmp[i][-8 : ]
        if suff in wav_suffs:
            wav_path_list.append(wav_path_list_tmp[i])
    wav_path_list.sort()
    
    # 直接依次处理4条音频
    for i in range(0, len(wav_path_list), 4):
        # 先检查一下4条音频的名字
        pref = wav_path_list[i][0 : -8]
        if pref != wav_path_list[i+1][0: -8] or pref != wav_path_list[i+2][0: -8] or pref != wav_path_list[i+3][0: -8]:
            print("name error! ", wav_path_list[i])
            exit()
        
        # step 2: 调用信号处理程序处理这4个输入
        wav_out_path = pref + '_out.wav'
        cmd = D_MICARRAY_EXE_PATH + " "
        for j in range(4):
            cmd = cmd + '\"' + wav_path_list[i + j] + '\"' + " "
        cmd = cmd + '\"' + wav_out_path + '\"'

        print(wav_out_path)
        print("")
        os.system(cmd)

            
# 切分文件，为了令文件名不冲突，可传入一个name_pref表示编号
# 不补0。最后一段不要了
def split_audio(audio_path, dst_dir, dur=2, name_pref='', SR=16000, padding=False):

    # 获取各级目录
    dir_list = audio_path.split('/')
    # 获取纯文件名及后缀
    purname, suff = os.path.splitext(dir_list[-1])
    
    x, _ = librosa.load(audio_path, SR, mono=True)
    
    pieces = (int)(x.shape[0] / (SR * dur))
    
    for j in range(pieces):
        dst_ff_name = name_pref + "_" + purname + "_" + str(j) + ".wav"
        dst_path = os.path.join(dst_dir, dst_ff_name)
        wave_data = x[j * SR * dur : (j + 1) * SR * dur] * 32768

        wave_data = wave_data.astype(np.short)
        f = wave.open(dst_path, "wb")
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        # 将wav_data转换为二进制数据写入文件
        f.writeframes(wave_data.tostring())
        f.close()


# 利用sox库降采样，原地操作
def sox_downsample(src):
    src_name = '\"' + str(src) + '\"'
    out_name = src_name.replace('.wav', '_tmp.wav')
    cmd0 = 'sox ' + src_name + ' -r 16000 ' + out_name
    os.system(cmd0)
    os.remove(src_name)
    os.rename(out_name, src_name)

# txt转wav
def txt2wav(txt_name, wav_name):
    import numpy as np
    import wave

    f = open(txt_name, "r")
    data = f.read().split()
    data_np = np.array(data).astype(np.short)

    wave_data = data_np.astype(np.short)
    f = wave.open(wav_name, "wb")
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)

    f.writeframes(wave_data.tostring())
    f.close()
