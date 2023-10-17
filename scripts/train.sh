python main.py --logdir 'outputs/512_codiff_id_mask_text'\
 --base 'configs/512_codiff_id_mask_text.yaml' -t --gpus 0,\
 -r '/home/jijunhao/diffusion/outputs/512_codiff_id_mask_text/2023-07-10T16-05-03_512_codiff_id_mask_text'

python main.py --logdir 'outputs/512_codiff_id_mask'\
 --base 'configs/512_codiff_id_mask.yaml' -t --gpus 0,\
 -r '/home/jijunhao/diffusion/outputs/512_codiff_id_mask/2023-07-16T13-29-40_512_codiff_id_mask'

# 50 epoch vae
# 166 epoch codiff_mask_text

# 117 epoch
python main.py --logdir 'outputs/512_id'\
 --base 'configs/512_id.yaml' -t --gpus 0, \
 -r '/home/jijunhao/diffusion/outputs/512_id/2023-07-01T21-06-39_512_id'

# 264 epoch
python main.py --logdir 'outputs/512_text'\
 --base 'configs/512_text.yaml' -t --gpus 0,\
 -r '/home/jijunhao/diffusion/outputs/512_text/2023-07-18T18-23-19_512_text'