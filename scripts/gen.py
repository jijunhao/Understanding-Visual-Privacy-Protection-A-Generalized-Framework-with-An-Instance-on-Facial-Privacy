"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/7/31 13:48
@Software: PyCharm 
@File : gen.py
"""
import os
import sys

# config_path = "configs/512_id.yaml"
# ckpt_path = "/home/jijunhao/diffusion/outputs/512_id/2023-07-01T21-06-39_512_id/pretrained/last.ckpt"
# save_folder = 'outputs/inference_512_id'

# # 遍历 1.jpg 到 1000.jpg 的文件名
# for i in [26,39,105,208,312,417,761,1080]:
#     img_path = f'datasets/image/image_512_downsampled_from_hq_1024/{i}.jpg'
#     mask_path = f'/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/{i}.png'
#     input_text = "The face is covered with short beard."
#     # 构建命令
#     command = f'python generate_512.py --init-img "{img_path}" --mask_path "{mask_path}"  --input_text "{input_text}" --condition 6'
#
#
#     try:
#         # 执行命令
#         os.system(command)
#     except KeyboardInterrupt:
#         # 捕获 KeyboardInterrupt 异常（Ctrl+C）
#         print("程序被终止")
#         sys.exit()

# c1 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/26.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/26.png" --input_text "This lady has no eyeglasses, and no smile. This person is in her thirties and has no bangs. This person is wearing lipstick. She has big lips, and arched eyebrows. She is wearing lipstick. This person has high cheekbones, and wavy hair. She is attractive and wears lipstick. The person has big lips, brown hair, arched eyebrows, and wavy hair. She has arched eyebrows, and big lips. The person has big lips, wavy hair, arched eyebrows, and high cheekbones and is wearing lipstick. She is young and wears heavy makeup. She wears lipstick. The woman has arched eyebrows, and big lips. She has big lips, wavy hair, high cheekbones, and arched eyebrows. She has brown hair, wavy hair, arched eyebrows, and big lips. She has high cheekbones, and arched eyebrows. She is young. She is attractive and is wearing heavy makeup. The woman wears heavy makeup. She is young and has high cheekbones, brown hair, and big lips. She is wearing heavy makeup. She wears heavy makeup. She is attractive." --condition 6'
# os.system(c1)
#
# c2 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "There is not any eyeglasses on his face and has short beard, and no fringe. This man is in the thirties. The full face is beamed with happiness. This man is smiling and has brown hair, big nose, and bushy eyebrows. He is young, and smiling. This man has sideburns. This smiling, and young person has narrow eyes, and big lips. He is smiling, and young. The person is young, and attractive and has bushy eyebrows, big lips, and big nose. The person has mouth slightly open, and brown hair. This man is attractive, and young and has big lips, sideburns, big nose, brown hair, and bushy eyebrows. The person is attractive and has sideburns, big nose, mustache, and bushy eyebrows. The man has mustache, and bushy eyebrows. The person is attractive, and smiling and has mustache, bushy eyebrows, big lips, brown hair, mouth slightly open, and narrow eyes. He has beard. He is young. The man has big lips, big nose, mustache, sideburns, narrow eyes, bushy eyebrows, and brown hair." --condition 6'
# os.system(c2)
#
# c3 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/208.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/208.png" --input_text "She looks serious with no smile in the face and has no eyeglasses, and long bangs. This woman is in her middle age. She is attractive and has brown hair, and high cheekbones. She is wearing lipstick, and heavy makeup. This attractive woman has bangs, and brown hair. The woman has high cheekbones and wears lipstick, and heavy makeup. This woman wears earrings. The person has brown hair. This attractive person has brown hair, and high cheekbones. She has high cheekbones. She wears heavy makeup. This person has brown hair, and high cheekbones and wears earrings. She is attractive. The woman has bangs, and high cheekbones. She wears earrings, and lipstick. She is attractive and wears heavy makeup. She is attractive and has brown hair. She has bangs, and brown hair. She wears earrings." --condition 6'
# os.system(c3)
#
# c4 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/1080.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1080.png" --input_text "This middle-aged gentleman has mustache of medium length, no smile, no bangs, and no eyeglasses. This man has pointy nose, and sideburns. The person has big nose. This chubby person has gray hair, bags under eyes, receding hairline, and goatee. This person is chubby and has sideburns, receding hairline, pointy nose, and gray hair. He is chubby. The man has receding hairline, and gray hair. He has receding hairline, goatee, bags under eyes, big nose, gray hair, and sideburns. This person has big nose, pointy nose, goatee, and sideburns. The man has goatee. He has bags under eyes, and gray hair. He has beard. This person has gray hair." --condition 6'
# os.system(c4)
#
# c5 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/4074.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/4074.png" --input_text "There is not any eyeglasses on the face. This female looks like an elderly and has no smile, and no bangs. She and is wearing lipstick has blond hair. She is wearing lipstick. This person and wears lipstick has gray hair, and blond hair. She has gray hair and wears lipstick. The woman has high cheekbones. She wears lipstick. The woman has wavy hair, high cheekbones, and blond hair and is wearing lipstick. The person is wearing lipstick. She has wavy hair, and gray hair. She has blond hair. This woman has gray hair. She has wavy hair, and blond hair. The woman has gray hair and is wearing lipstick." --condition 6'
# os.system(c5)
#
# c6 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/3982.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3982.png" --input_text "This guy has no fringe, no smile, and no mustache at all. He is in the eighties and has thin frame eyeglasses. He has gray hair. He is wearing necktie. The person has bags under eyes, eyeglasses, and big nose. He is bald and has gray hair, and receding hairline. The man is wearing necktie. He has eyeglasses, and receding hairline. He has no beard. The person has receding hairline, big nose, gray hair, and bags under eyes and is wearing necktie. The person has bags under eyes, big nose, gray hair, and receding hairline and is wearing necktie. He is bald and is wearing necktie. He is chubby. He is bald. He has receding hairline, and big nose. This man is wearing necktie. The person has big nose, receding hairline, and bags under eyes and wears necktie. This person is wearing necktie. He has eyeglasses, big nose, and receding hairline." --condition 6'
# os.system(c6)
#
# c7 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/2570.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2570.png" --input_text "He is wearing eyeglasses with thin frame and has no mustache, no smile, and no fringe. He looks very young. The man has bags under eyes, and bushy eyebrows. The man is attractive and has bushy eyebrows, and eyeglasses. He has no beard. The man has bushy eyebrows, and eyeglasses. This man has bags under eyes. This person has bushy eyebrows, and bags under eyes. This man has eyeglasses. This person has eyeglasses, and bushy eyebrows. The person has bags under eyes, and bushy eyebrows. This person is attractive and has eyeglasses. He is attractive. He is young. This person has eyeglasses, and bags under eyes." --condition 6'
# os.system(c7)
#
# c8 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/3017.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3017.png" --input_text "This lady is wearing sunglasses with thin frame and has no smile, and no fringe. This person is in the thirties. She has blond hair, eyeglasses, and straight hair and wears necklace. This woman is young and has blond hair, and eyeglasses. The person is wearing necklace. She has eyeglasses, and receding hairline. She is young and wears necklace. This person has receding hairline, and straight hair. She has eyeglasses, straight hair, and receding hairline and is wearing necklace. She has straight hair, blond hair, and eyeglasses. She is wearing necklace. She has receding hairline. The person has blond hair, receding hairline, and straight hair and wears necklace. The woman is young and has eyeglasses, and receding hairline. The woman has blond hair. She is young." --condition 6'
# os.system(c8)

# c1 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/0.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/26.png" --input_text "This lady has no eyeglasses, and no smile. This person is in her thirties and has no bangs. This person is wearing lipstick. She has big lips, and arched eyebrows. She is wearing lipstick. This person has high cheekbones, and wavy hair. She is attractive and wears lipstick. The person has big lips, brown hair, arched eyebrows, and wavy hair. She has arched eyebrows, and big lips. The person has big lips, wavy hair, arched eyebrows, and high cheekbones and is wearing lipstick. She is young and wears heavy makeup. She wears lipstick. The woman has arched eyebrows, and big lips. She has big lips, wavy hair, high cheekbones, and arched eyebrows. She has brown hair, wavy hair, arched eyebrows, and big lips. She has high cheekbones, and arched eyebrows. She is young. She is attractive and is wearing heavy makeup. The woman wears heavy makeup. She is young and has high cheekbones, brown hair, and big lips. She is wearing heavy makeup. She wears heavy makeup. She is attractive." --condition 6'
# os.system(c1)
#
# c2 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/0.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "There is not any eyeglasses on his face and has short beard, and no fringe. This man is in the thirties. The full face is beamed with happiness. This man is smiling and has brown hair, big nose, and bushy eyebrows. He is young, and smiling. This man has sideburns. This smiling, and young person has narrow eyes, and big lips. He is smiling, and young. The person is young, and attractive and has bushy eyebrows, big lips, and big nose. The person has mouth slightly open, and brown hair. This man is attractive, and young and has big lips, sideburns, big nose, brown hair, and bushy eyebrows. The person is attractive and has sideburns, big nose, mustache, and bushy eyebrows. The man has mustache, and bushy eyebrows. The person is attractive, and smiling and has mustache, bushy eyebrows, big lips, brown hair, mouth slightly open, and narrow eyes. He has beard. He is young. The man has big lips, big nose, mustache, sideburns, narrow eyes, bushy eyebrows, and brown hair." --condition 6'
# os.system(c2)
#
# c3 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/0.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/208.png" --input_text "She looks serious with no smile in the face and has no eyeglasses, and long bangs. This woman is in her middle age. She is attractive and has brown hair, and high cheekbones. She is wearing lipstick, and heavy makeup. This attractive woman has bangs, and brown hair. The woman has high cheekbones and wears lipstick, and heavy makeup. This woman wears earrings. The person has brown hair. This attractive person has brown hair, and high cheekbones. She has high cheekbones. She wears heavy makeup. This person has brown hair, and high cheekbones and wears earrings. She is attractive. The woman has bangs, and high cheekbones. She wears earrings, and lipstick. She is attractive and wears heavy makeup. She is attractive and has brown hair. She has bangs, and brown hair. She wears earrings." --condition 6'
# os.system(c3)
#
# c4 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/0.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1080.png" --input_text "This middle-aged gentleman has mustache of medium length, no smile, no bangs, and no eyeglasses. This man has pointy nose, and sideburns. The person has big nose. This chubby person has gray hair, bags under eyes, receding hairline, and goatee. This person is chubby and has sideburns, receding hairline, pointy nose, and gray hair. He is chubby. The man has receding hairline, and gray hair. He has receding hairline, goatee, bags under eyes, big nose, gray hair, and sideburns. This person has big nose, pointy nose, goatee, and sideburns. The man has goatee. He has bags under eyes, and gray hair. He has beard. This person has gray hair." --condition 6'
# os.system(c4)
#
# c5 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/0.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/4074.png" --input_text "There is not any eyeglasses on the face. This female looks like an elderly and has no smile, and no bangs. She and is wearing lipstick has blond hair. She is wearing lipstick. This person and wears lipstick has gray hair, and blond hair. She has gray hair and wears lipstick. The woman has high cheekbones. She wears lipstick. The woman has wavy hair, high cheekbones, and blond hair and is wearing lipstick. The person is wearing lipstick. She has wavy hair, and gray hair. She has blond hair. This woman has gray hair. She has wavy hair, and blond hair. The woman has gray hair and is wearing lipstick." --condition 6'
# os.system(c5)
#
# c6 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/0.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3982.png" --input_text "This guy has no fringe, no smile, and no mustache at all. He is in the eighties and has thin frame eyeglasses. He has gray hair. He is wearing necktie. The person has bags under eyes, eyeglasses, and big nose. He is bald and has gray hair, and receding hairline. The man is wearing necktie. He has eyeglasses, and receding hairline. He has no beard. The person has receding hairline, big nose, gray hair, and bags under eyes and is wearing necktie. The person has bags under eyes, big nose, gray hair, and receding hairline and is wearing necktie. He is bald and is wearing necktie. He is chubby. He is bald. He has receding hairline, and big nose. This man is wearing necktie. The person has big nose, receding hairline, and bags under eyes and wears necktie. This person is wearing necktie. He has eyeglasses, big nose, and receding hairline." --condition 6'
# os.system(c6)
#
# c7 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/0.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2570.png" --input_text "He is wearing eyeglasses with thin frame and has no mustache, no smile, and no fringe. He looks very young. The man has bags under eyes, and bushy eyebrows. The man is attractive and has bushy eyebrows, and eyeglasses. He has no beard. The man has bushy eyebrows, and eyeglasses. This man has bags under eyes. This person has bushy eyebrows, and bags under eyes. This man has eyeglasses. This person has eyeglasses, and bushy eyebrows. The person has bags under eyes, and bushy eyebrows. This person is attractive and has eyeglasses. He is attractive. He is young. This person has eyeglasses, and bags under eyes." --condition 6'
# os.system(c7)
#
# c8 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/0.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3017.png" --input_text "This lady is wearing sunglasses with thin frame and has no smile, and no fringe. This person is in the thirties. She has blond hair, eyeglasses, and straight hair and wears necklace. This woman is young and has blond hair, and eyeglasses. The person is wearing necklace. She has eyeglasses, and receding hairline. She is young and wears necklace. This person has receding hairline, and straight hair. She has eyeglasses, straight hair, and receding hairline and is wearing necklace. She has straight hair, blond hair, and eyeglasses. She is wearing necklace. She has receding hairline. The person has blond hair, receding hairline, and straight hair and wears necklace. The woman is young and has eyeglasses, and receding hairline. The woman has blond hair. She is young." --condition 6'
# os.system(c8)
#
#
# c1 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/18.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/26.png" --input_text "This lady has no eyeglasses, and no smile. This person is in her thirties and has no bangs. This person is wearing lipstick. She has big lips, and arched eyebrows. She is wearing lipstick. This person has high cheekbones, and wavy hair. She is attractive and wears lipstick. The person has big lips, brown hair, arched eyebrows, and wavy hair. She has arched eyebrows, and big lips. The person has big lips, wavy hair, arched eyebrows, and high cheekbones and is wearing lipstick. She is young and wears heavy makeup. She wears lipstick. The woman has arched eyebrows, and big lips. She has big lips, wavy hair, high cheekbones, and arched eyebrows. She has brown hair, wavy hair, arched eyebrows, and big lips. She has high cheekbones, and arched eyebrows. She is young. She is attractive and is wearing heavy makeup. The woman wears heavy makeup. She is young and has high cheekbones, brown hair, and big lips. She is wearing heavy makeup. She wears heavy makeup. She is attractive." --condition 6'
# os.system(c1)
#
# c2 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/18.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "There is not any eyeglasses on his face and has short beard, and no fringe. This man is in the thirties. The full face is beamed with happiness. This man is smiling and has brown hair, big nose, and bushy eyebrows. He is young, and smiling. This man has sideburns. This smiling, and young person has narrow eyes, and big lips. He is smiling, and young. The person is young, and attractive and has bushy eyebrows, big lips, and big nose. The person has mouth slightly open, and brown hair. This man is attractive, and young and has big lips, sideburns, big nose, brown hair, and bushy eyebrows. The person is attractive and has sideburns, big nose, mustache, and bushy eyebrows. The man has mustache, and bushy eyebrows. The person is attractive, and smiling and has mustache, bushy eyebrows, big lips, brown hair, mouth slightly open, and narrow eyes. He has beard. He is young. The man has big lips, big nose, mustache, sideburns, narrow eyes, bushy eyebrows, and brown hair." --condition 6'
# os.system(c2)
#
# c3 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/18.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/208.png" --input_text "She looks serious with no smile in the face and has no eyeglasses, and long bangs. This woman is in her middle age. She is attractive and has brown hair, and high cheekbones. She is wearing lipstick, and heavy makeup. This attractive woman has bangs, and brown hair. The woman has high cheekbones and wears lipstick, and heavy makeup. This woman wears earrings. The person has brown hair. This attractive person has brown hair, and high cheekbones. She has high cheekbones. She wears heavy makeup. This person has brown hair, and high cheekbones and wears earrings. She is attractive. The woman has bangs, and high cheekbones. She wears earrings, and lipstick. She is attractive and wears heavy makeup. She is attractive and has brown hair. She has bangs, and brown hair. She wears earrings." --condition 6'
# os.system(c3)
#
# c4 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/18.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1080.png" --input_text "This middle-aged gentleman has mustache of medium length, no smile, no bangs, and no eyeglasses. This man has pointy nose, and sideburns. The person has big nose. This chubby person has gray hair, bags under eyes, receding hairline, and goatee. This person is chubby and has sideburns, receding hairline, pointy nose, and gray hair. He is chubby. The man has receding hairline, and gray hair. He has receding hairline, goatee, bags under eyes, big nose, gray hair, and sideburns. This person has big nose, pointy nose, goatee, and sideburns. The man has goatee. He has bags under eyes, and gray hair. He has beard. This person has gray hair." --condition 6'
# os.system(c4)
#
# c5 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/18.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/4074.png" --input_text "There is not any eyeglasses on the face. This female looks like an elderly and has no smile, and no bangs. She and is wearing lipstick has blond hair. She is wearing lipstick. This person and wears lipstick has gray hair, and blond hair. She has gray hair and wears lipstick. The woman has high cheekbones. She wears lipstick. The woman has wavy hair, high cheekbones, and blond hair and is wearing lipstick. The person is wearing lipstick. She has wavy hair, and gray hair. She has blond hair. This woman has gray hair. She has wavy hair, and blond hair. The woman has gray hair and is wearing lipstick." --condition 6'
# os.system(c5)
#
# c6 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/18.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3982.png" --input_text "This guy has no fringe, no smile, and no mustache at all. He is in the eighties and has thin frame eyeglasses. He has gray hair. He is wearing necktie. The person has bags under eyes, eyeglasses, and big nose. He is bald and has gray hair, and receding hairline. The man is wearing necktie. He has eyeglasses, and receding hairline. He has no beard. The person has receding hairline, big nose, gray hair, and bags under eyes and is wearing necktie. The person has bags under eyes, big nose, gray hair, and receding hairline and is wearing necktie. He is bald and is wearing necktie. He is chubby. He is bald. He has receding hairline, and big nose. This man is wearing necktie. The person has big nose, receding hairline, and bags under eyes and wears necktie. This person is wearing necktie. He has eyeglasses, big nose, and receding hairline." --condition 6'
# os.system(c6)
#
# c7 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/18.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2570.png" --input_text "He is wearing eyeglasses with thin frame and has no mustache, no smile, and no fringe. He looks very young. The man has bags under eyes, and bushy eyebrows. The man is attractive and has bushy eyebrows, and eyeglasses. He has no beard. The man has bushy eyebrows, and eyeglasses. This man has bags under eyes. This person has bushy eyebrows, and bags under eyes. This man has eyeglasses. This person has eyeglasses, and bushy eyebrows. The person has bags under eyes, and bushy eyebrows. This person is attractive and has eyeglasses. He is attractive. He is young. This person has eyeglasses, and bags under eyes." --condition 6'
# os.system(c7)
#
# c8 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/18.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3017.png" --input_text "This lady is wearing sunglasses with thin frame and has no smile, and no fringe. This person is in the thirties. She has blond hair, eyeglasses, and straight hair and wears necklace. This woman is young and has blond hair, and eyeglasses. The person is wearing necklace. She has eyeglasses, and receding hairline. She is young and wears necklace. This person has receding hairline, and straight hair. She has eyeglasses, straight hair, and receding hairline and is wearing necklace. She has straight hair, blond hair, and eyeglasses. She is wearing necklace. She has receding hairline. The person has blond hair, receding hairline, and straight hair and wears necklace. The woman is young and has eyeglasses, and receding hairline. The woman has blond hair. She is young." --condition 6'
# os.system(c8)
#
# c1 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/419.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/26.png" --input_text "This lady has no eyeglasses, and no smile. This person is in her thirties and has no bangs. This person is wearing lipstick. She has big lips, and arched eyebrows. She is wearing lipstick. This person has high cheekbones, and wavy hair. She is attractive and wears lipstick. The person has big lips, brown hair, arched eyebrows, and wavy hair. She has arched eyebrows, and big lips. The person has big lips, wavy hair, arched eyebrows, and high cheekbones and is wearing lipstick. She is young and wears heavy makeup. She wears lipstick. The woman has arched eyebrows, and big lips. She has big lips, wavy hair, high cheekbones, and arched eyebrows. She has brown hair, wavy hair, arched eyebrows, and big lips. She has high cheekbones, and arched eyebrows. She is young. She is attractive and is wearing heavy makeup. The woman wears heavy makeup. She is young and has high cheekbones, brown hair, and big lips. She is wearing heavy makeup. She wears heavy makeup. She is attractive." --condition 6'
# os.system(c1)
#
# c2 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/419.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "There is not any eyeglasses on his face and has short beard, and no fringe. This man is in the thirties. The full face is beamed with happiness. This man is smiling and has brown hair, big nose, and bushy eyebrows. He is young, and smiling. This man has sideburns. This smiling, and young person has narrow eyes, and big lips. He is smiling, and young. The person is young, and attractive and has bushy eyebrows, big lips, and big nose. The person has mouth slightly open, and brown hair. This man is attractive, and young and has big lips, sideburns, big nose, brown hair, and bushy eyebrows. The person is attractive and has sideburns, big nose, mustache, and bushy eyebrows. The man has mustache, and bushy eyebrows. The person is attractive, and smiling and has mustache, bushy eyebrows, big lips, brown hair, mouth slightly open, and narrow eyes. He has beard. He is young. The man has big lips, big nose, mustache, sideburns, narrow eyes, bushy eyebrows, and brown hair." --condition 6'
# os.system(c2)
#
# c3 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/419.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/208.png" --input_text "She looks serious with no smile in the face and has no eyeglasses, and long bangs. This woman is in her middle age. She is attractive and has brown hair, and high cheekbones. She is wearing lipstick, and heavy makeup. This attractive woman has bangs, and brown hair. The woman has high cheekbones and wears lipstick, and heavy makeup. This woman wears earrings. The person has brown hair. This attractive person has brown hair, and high cheekbones. She has high cheekbones. She wears heavy makeup. This person has brown hair, and high cheekbones and wears earrings. She is attractive. The woman has bangs, and high cheekbones. She wears earrings, and lipstick. She is attractive and wears heavy makeup. She is attractive and has brown hair. She has bangs, and brown hair. She wears earrings." --condition 6'
# os.system(c3)
#
# c4 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/419.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1080.png" --input_text "This middle-aged gentleman has mustache of medium length, no smile, no bangs, and no eyeglasses. This man has pointy nose, and sideburns. The person has big nose. This chubby person has gray hair, bags under eyes, receding hairline, and goatee. This person is chubby and has sideburns, receding hairline, pointy nose, and gray hair. He is chubby. The man has receding hairline, and gray hair. He has receding hairline, goatee, bags under eyes, big nose, gray hair, and sideburns. This person has big nose, pointy nose, goatee, and sideburns. The man has goatee. He has bags under eyes, and gray hair. He has beard. This person has gray hair." --condition 6'
# os.system(c4)
#
# c5 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/419.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/4074.png" --input_text "There is not any eyeglasses on the face. This female looks like an elderly and has no smile, and no bangs. She and is wearing lipstick has blond hair. She is wearing lipstick. This person and wears lipstick has gray hair, and blond hair. She has gray hair and wears lipstick. The woman has high cheekbones. She wears lipstick. The woman has wavy hair, high cheekbones, and blond hair and is wearing lipstick. The person is wearing lipstick. She has wavy hair, and gray hair. She has blond hair. This woman has gray hair. She has wavy hair, and blond hair. The woman has gray hair and is wearing lipstick." --condition 6'
# os.system(c5)
#
# c6 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/419.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3982.png" --input_text "This guy has no fringe, no smile, and no mustache at all. He is in the eighties and has thin frame eyeglasses. He has gray hair. He is wearing necktie. The person has bags under eyes, eyeglasses, and big nose. He is bald and has gray hair, and receding hairline. The man is wearing necktie. He has eyeglasses, and receding hairline. He has no beard. The person has receding hairline, big nose, gray hair, and bags under eyes and is wearing necktie. The person has bags under eyes, big nose, gray hair, and receding hairline and is wearing necktie. He is bald and is wearing necktie. He is chubby. He is bald. He has receding hairline, and big nose. This man is wearing necktie. The person has big nose, receding hairline, and bags under eyes and wears necktie. This person is wearing necktie. He has eyeglasses, big nose, and receding hairline." --condition 6'
# os.system(c6)
#
# c7 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/419.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2570.png" --input_text "He is wearing eyeglasses with thin frame and has no mustache, no smile, and no fringe. He looks very young. The man has bags under eyes, and bushy eyebrows. The man is attractive and has bushy eyebrows, and eyeglasses. He has no beard. The man has bushy eyebrows, and eyeglasses. This man has bags under eyes. This person has bushy eyebrows, and bags under eyes. This man has eyeglasses. This person has eyeglasses, and bushy eyebrows. The person has bags under eyes, and bushy eyebrows. This person is attractive and has eyeglasses. He is attractive. He is young. This person has eyeglasses, and bags under eyes." --condition 6'
# os.system(c7)
#
# c8 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/419.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3017.png" --input_text "This lady is wearing sunglasses with thin frame and has no smile, and no fringe. This person is in the thirties. She has blond hair, eyeglasses, and straight hair and wears necklace. This woman is young and has blond hair, and eyeglasses. The person is wearing necklace. She has eyeglasses, and receding hairline. She is young and wears necklace. This person has receding hairline, and straight hair. She has eyeglasses, straight hair, and receding hairline and is wearing necklace. She has straight hair, blond hair, and eyeglasses. She is wearing necklace. She has receding hairline. The person has blond hair, receding hairline, and straight hair and wears necklace. The woman is young and has eyeglasses, and receding hairline. The woman has blond hair. She is young." --condition 6'
# os.system(c8)

# c1 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/1377.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1377.png" --input_text "no beard" --condition 6'
# os.system(c1)
#
#
# c2 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "no beard" --condition 6'
# os.system(c2)
#
# c3 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/2217.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2217.png" --input_text "no beard" --condition 6'
# os.system(c3)
#
# c4 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/1080.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1080.png" --input_text "no beard" --condition 6'
# os.system(c4)
#
# c5 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/4074.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/4074.png" --input_text "with short beard" --condition 6'
# os.system(c5)
#
# c6 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/3982.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3982.png" --input_text "with short beard" --condition 6'
# os.system(c6)
#
# c7 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/2570.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2570.png" --input_text "with short beard" --condition 6'
# os.system(c7)
#
# c8 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/3017.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3017.png" --input_text "with short beard" --condition 6'
# os.system(c8)


# c1 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/1377.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1377.png" --input_text "no beard" --condition 6'
# os.system(c1)
#
#
# c2 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "no beard" --condition 6'
# os.system(c2)
#
# c3 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/2217.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2217.png" --input_text "no beard" --condition 6'
# os.system(c3)
#
# c4 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/1080.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1080.png" --input_text "no beard" --condition 6'
# os.system(c4)
#
# c5 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/4074.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/4074.png" --input_text "with short beard" --condition 6'
# os.system(c5)
#
# c6 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/3982.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3982.png" --input_text "with short beard" --condition 6'
# os.system(c6)
#
# c7 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/2570.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2570.png" --input_text "with short beard" --condition 6'
# os.system(c7)
#
# c8 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/3017.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3017.png" --input_text "with short beard" --condition 6'
# os.system(c8)

# c1 = 'python generate_512.py --save_folder "outputs/smile" --init-img "datasets/image/image_512_downsampled_from_hq_1024/1377.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1377.png" --input_text "smile with the teeth visible" --condition 6'
# os.system(c1)
#
#
# c2 = 'python generate_512.py --save_folder "outputs/smile" --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "smile with the teeth visible" --condition 6'
# os.system(c2)
#
# c3 = 'python generate_512.py --save_folder "outputs/smile" --init-img "datasets/image/image_512_downsampled_from_hq_1024/2217.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2217.png" --input_text "smile with the teeth visible" --condition 6'
# os.system(c3)
#
# c4 = 'python generate_512.py --save_folder "outputs/smile" --init-img "datasets/image/image_512_downsampled_from_hq_1024/1080.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/1080.png" --input_text "smile with the teeth visible" --condition 6'
# os.system(c4)
#
# c5 = 'python generate_512.py --save_folder "outputs/smile" --init-img "datasets/image/image_512_downsampled_from_hq_1024/4074.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/4074.png" --input_text "no smile" --condition 6'
# os.system(c5)
#
# c6 = 'python generate_512.py --save_folder "outputs/smile" --init-img "datasets/image/image_512_downsampled_from_hq_1024/3982.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3982.png" --input_text "smile with the teeth visible" --condition 6'
# os.system(c6)
#
# c7 = 'python generate_512.py --save_folder "outputs/smile" --init-img "datasets/image/image_512_downsampled_from_hq_1024/2570.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/2570.png" --input_text "no smile" --condition 6'
# os.system(c7)
#
# c8 = 'python generate_512.py --save_folder "outputs/smile" --init-img "datasets/image/image_512_downsampled_from_hq_1024/3017.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/3017.png" --input_text "no smile" --condition 6'
# os.system(c8)





# c2 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "no beard" --condition 6'
# os.system(c2)
#
# c2 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "no beard" --condition 6'
# os.system(c2)
#
# c2 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "no beard" --condition 6'
# os.system(c2)
#
# c2 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "no beard" --condition 6'
# os.system(c2)
#
# c2 = 'python generate_512.py --save_folder "outputs/att" --init-img "datasets/image/image_512_downsampled_from_hq_1024/39.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/39.png" --input_text "no beard" --condition 6'
# os.system(c2)

c1 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/4665.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/4665.png" --input_text "This person has straight hair, pointy nose, high cheekbones, oval face, and mouth slightly open and is wearing earrings. This attractive, and young person has bangs, pointy nose, and rosy cheeks. She wears earrings. The person wears necklace, earrings. She has pointy nose, oval face, straight hair, mouth slightly open, high cheekbones, and rosy cheeks. The woman has rosy cheeks. She is wearing heavy makeup, earrings, and necklace. She has straight hair, high cheekbones, mouth slightly open, pointy nose, and bangs. The person is wearing necklace. She has oval face, rosy cheeks, straight hair, and pointy nose and is wearing lipstick. She has pointy nose, and high cheekbones. She has mouth slightly open, rosy cheeks, pointy nose, and straight hair. She is young, and attractive. The person is young, and attractive and has rosy cheeks. She has straight hair, mouth slightly open, rosy cheeks, and bangs. She is smiling, and young and wears heavy makeup, and lipstick. She is attractive, and smiling. She is attractive and wears heavy makeup, and earrings. She is attractive." --condition 6'
os.system(c1)

c2 = 'python generate_512.py --init-img "datasets/image/image_512_downsampled_from_hq_1024/15123.jpg" --mask_path "/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/15123.png" --input_text "This man has bushy eyebrows, high cheekbones, mouth slightly open, bags under eyes, and black hair. This person has high cheekbones, big lips, straight hair, bushy eyebrows, mouth slightly open, and bags under eyes. This person has high cheekbones, black hair, bushy eyebrows, bags under eyes, straight hair, and big lips. This person is young, and smiling and has mouth slightly open, black hair, and big lips. The person is young, and smiling and has big lips, and bushy eyebrows. He has mouth slightly open, straight hair, and big lips. The man is attractive, and young and has bushy eyebrows, big lips, straight hair, high cheekbones, and mouth slightly open. This person has big lips, black hair, high cheekbones, mouth slightly open, bushy eyebrows, and straight hair. The man has mouth slightly open, and bags under eyes. He is attractive. He has beard. The person has bags under eyes, and black hair. He is smiling." --condition 6'
os.system(c2)