name=VLNBERT-test-GEO

flag="--vlnbert prevalent

      --submit 0
      --test_only 0
      --load snap/VLNBERT-train-GEO/state_dict/best_val_unseen

      --train validlistener

      --features clipresnet50_4
      --maxAction 15
      --batchSize 12
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --feature_size 640
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5

      --slot_attn 
      --slot_dropout 0.7 
      --slot_residual 
      --slot_local_mask
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python r2r_src/train.py $flag --name $name