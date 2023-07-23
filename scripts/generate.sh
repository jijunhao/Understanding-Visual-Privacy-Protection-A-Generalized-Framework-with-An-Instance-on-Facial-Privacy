# text+mask+id
python generate_512.py --init-img 'datasets/image/image_512_downsampled_from_hq_1024/29980.jpg' \
 --mask_path '/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/29980.png' \
 --input_text "This man is in the fifties. The face is covered with short beard." \
 --condition 6

# id
python generate_512.py --config_path "configs/512_id.yaml"\
 --ckpt_path "/home/jijunhao/diffusion/outputs/512_id/2023-07-01T21-06-39_512_id/pretrained/last.ckpt"\
 --init-img 'datasets/image/image_512_downsampled_from_hq_1024/29980.jpg' \
 --save_folder 'outputs/inference_512_id' \
 --condition 2

# mask
python generate_512.py --config_path "configs/512_mask.yaml"\
 --ckpt_path "pretrained/512_mask.ckpt"\
 --init-img 'datasets/image/image_512_downsampled_from_hq_1024/29980.jpg' \
 --mask_path '/home/jijunhao/diffusion/data/CelebAMask-HQ/CelebAMask-HQ-mask-color-palette/29980.png' \
 --save_folder 'outputs/inference_512_mask' \
 --condition 1

# text
python generate_512.py --config_path "configs/512_text.yaml"\
 --ckpt_path "/home/jijunhao/diffusion/outputs/512_text/2023-07-18T18-23-19_512_text/pretrained/last.ckpt"\
 --init-img 'datasets/image/image_512_downsampled_from_hq_1024/29980.jpg' \
 --input_text "There is no beard This woman looks extremely young." \
 --save_folder 'outputs/inference_512_text' \
 --condition 0

