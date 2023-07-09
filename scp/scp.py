import os

base_path = 'A:\WORK\project\VOICE_DATA\TIMIT\TRAIN\DR4'
with open('M:/DCCRN/scp/train.scp', 'a+', encoding='utf-8') as f:
# base_path = 'd:/Users/J/Desktop/voice/dataset/TIMIT/data/TRAIN/'
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_name = os.path.join(root,file)

            if file_name.endswith('.WAV'):
                print(file_name)
                f.write('%s\n' % file_name)
f.close()

# base2_path = 'A:\WORK\project\VOICE_DATA\TIMIT\TEST\DR4'
# with open('M:/DCCRN/scp/test.scp','a+',encoding='utf-8') as f:
# # base_path = 'd:/Users/J/Desktop/voice/dataset/TIMIT/data/TRAIN/'
#     for root,dirs,files in os.walk(base2_path):
#         for file in files:
#             file_name = os.path.join(root,file)
#
#             if file_name.endswith('.WAV'):
#                 print(file_name)
#                 f.write('%s\n' % file_name)
# f.close()