#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021.02.04

@author: chenyingbo
"""

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
