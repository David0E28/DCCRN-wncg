import os

base_path = 'A:\WORK\project\VOICE_DATA\TIMIT\TRAIN\DR1'
with open('M:/DCCRN/scp/train.scp','w+',encoding='utf-8') as f:
# base_path = 'd:/Users/J/Desktop/voice/dataset/TIMIT/data/TRAIN/'
    for root,dirs,files in os.walk(base_path):
        for file in files:
            file_name = os.path.join(root,file)

            if file_name.endswith('.WAV'):
                print(file_name)
                f.write('%s\n' % file_name)
f.close()