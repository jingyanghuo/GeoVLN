export PYTHONPATH=$PYTHONPATH:/sdc1/huojingyang/proj/VLN/VLN-HAMT-final/finetune_src

flag="--history_fusion 
      --features vitbase_r2rfte2e 
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json 
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt 
      --root_dir ../datasets 
      --output_dir ../datasets/R2R/trained_models/Geo-HAMT-TRAIN 

      --dataset r2r 
      --ob_type pano 
      --world_size 1 
      --seed 0 
      --num_l_layers 9 
      --num_x_layers 4 
      --hist_enc_pano 
      --hist_pano_num_layers 2 
      --fix_lang_embedding 
      --fix_hist_embedding 
      --feedback sample 
      --max_action_len 15 
      --max_instr_len 60 
      --image_feat_size 768 
      --angle_feat_size 4 

      --lr 1e-5 
      --iters 300000 
      --log_every 2000 

      --batch_size 12 
      --optim adamW 

      --ml_weight 0.2 
      --feat_dropout 0.4 
      --dropout 0.5 

      --slot_attn 
      --slot_dropout 0.5 
      --slot_residual 
      --slot_local_mask 
      "


# training
# CUDA_VISIBLE_DEVICES='1' python r2r_geo_slot/main.py $flag --eval_first


# inference
CUDA_VISIBLE_DEVICES='1' python r2r_geo_slot/main.py  $flag \
      --resume_file ../datasets/R2R/trained_models/Geo-HAMT-TRAIN/best_val_unseen \
      --test
