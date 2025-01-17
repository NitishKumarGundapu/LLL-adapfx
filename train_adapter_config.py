
'''
nohup python train_adapter_gen_mam_config.py 
    --data_dir data 
    --model_dir_root new_models 
    --seq_train_type lll 
    --model_name gpt2 
    --n_gpus 8 
    --n_workers 75 
    --fp32 
    --n_train_epochs 8 
    --gen_lm_sample_percentage 0.2 
    --tasks amazon ag yahoo yelp 
    --lm_lambda 0.2 
    > Trainlogs_AdaptGen_4cls_mam_config.out
'''
'''nohup python train_adapter_gen_mam_config.py 
    --data_dir data --model_dir_root models 
    --seq_train_type lll --model_name gpt2 
    --n_gpus 10 
    --n_workers 75 
    --fp32 
    --n_train_epochs 8 
    --gen_lm_sample_percentage 0.2 
    --tasks sst srl woz.en 
    --lm_lambda 0.2 >
    Trainlogs_adaptgen_3og_mam_config.out
    

nohup python test_adapter_from_scratch.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 35 --n_train_epochs 10 --gen_lm_sample_percentage 0.1 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_testing_logs/Testlogs_Adaptgen_3og.out

nohup python train_adapter_gen_mam_config_2.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 12 --n_workers 75 --fp32 --n_train_epochs 10 --gen_lm_sample_percentage 0.1 --tasks sst srl woz.en --lm_lambda 0.2 > /raid/amana/Lamol_with_adaptergen/nitish_training_logs/Trainlogs_adaptgen_3_og_mam_config.out
'''

'''

adapter_config = ConfigUnion(
                            PfeifferConfig(mh_adapter=True, output_adapter=False, reduction_factor=args.rf, non_linearity="relu",leave_out = args.leaveout),
                            PfeifferConfig(mh_adapter=False, output_adapter=True, reduction_factor=args.rf, non_linearity="relu",leave_out = args.leaveout)
                        )

'''


import torch
import time
from torch.utils.data import DataLoader
from torch import nn
from transformers import AdamW, WEIGHTS_NAME, get_linear_schedule_with_warmup

import csv 
import numpy as np
import os
import random
import logging
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict
from utils import *
from settings import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
from transformers.adapters import ConfigUnion, AdapterConfig, PrefixTuningConfig
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")


# def set_seed(seed: int = 42) -> None:
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # When running on the CuDNN backend, two further options must be set
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # Set a fixed value for the hash seed
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     print(f"Random seed set as {seed}")


def train(task_ids, model,train_type=0):
    tasks = [args.tasks[task_id] for task_id in task_ids]
    test_run = False

    logger.info("start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)
    print(task_ids)
    

    train_dataset = [TASK_DICT[t]["train"] for t in tasks]
    train_extra_data = []
    if train_type==0:
        if "lll" in args.seq_train_type and task_ids[0] > 0 and not args.skip_tasks:
            prev_task = args.tasks[task_ids[0]-1]
            with torch.no_grad():
                create_extra_data(tasks[0], prev_task, model, train_extra_data,test_run)
        elif "gem" in args.seq_train_type and task_ids[0] > 0: 
            get_real_data(tasks[0], train_extra_data, accum=False, encode=True)
            args.memory_data.append(train_extra_data)
            train_extra_data = []
    logger.info('extra training data size: {}'.format(len(train_extra_data)))

    if not model:
        # which_model_to_load = model_dir if os.path.isfile(os.path.join(model_dir, FINAL_SAVE_NAME)) else args.model_name
        model = MODEL_CLASS.from_pretrained(args.model_name).cuda()
        model.resize_token_embeddings(len(TOKENIZER))
        if not args.fp32:
            model = FP16_Module(model)

    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, torch.ones([1]).cuda()))
        
    ##adding adapters
    
    if task_ids[0]>0 and train_type==1:
        adapter_name = args.tasks[task_ids[0]].replace(".", "_")
        # if args.flat == False:
        #     adapter_config = ConfigUnion(
        #                     PrefixTuningConfig(cross_prefix=True, prefix_length=args.prefixlength, bottleneck_size=args.bottle_neck_size,non_linearity='relu',leave_out=args.leaveout),
        #                     AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=args.rf, non_linearity="relu", init_weights="mam_adapter", leave_out=args.leaveout),
        #                 )
        # else:
        #     adapter_config = ConfigUnion(
        #                     PrefixTuningConfig(cross_prefix=True, prefix_length=args.prefixlength, flat = True,non_linearity='relu',leave_out=args.leaveout),
        #                     AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=args.rf, non_linearity="relu", init_weights="mam_adapter", leave_out=args.leaveout),
        #                 )
        adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=args.rf, non_linearity="relu", init_weights="mam_adapter", leave_out=args.leaveout)
        model.add_adapter(adapter_name, config=adapter_config)
        print(model.adapter_summary())
        model.set_active_adapters(adapter_name)
        model = model.to(args.device_ids[0])
        model.train_adapter(adapter_name)

    if args.skip_tasks and len(tasks) == 1:
        logger.info("*********** skip task: {} ***********".format(tasks[0]))
        if tasks[0] in args.skip_tasks:
            if len(args.skip_tasks) == 1:
                model_dir = get_model_dir(tasks)
                model_path = os.path.join(model_dir, FINAL_SAVE_NAME)
                config_path = os.path.join(model_dir,CONFIG_NAME)
                model_config = CONFIG_CLASS.from_json_file(config_path)
                model = MODEL_CLASS(model_config).cuda()
                ## Addtion
                prev_task_adapter = args.tasks[task_ids[0]-1]
                adapter_name = args.tasks[task_ids[0]-1].replace(".", "_") ##Change
                adapter_name = args.tasks[0].replace(".", "_")
                adapter_dir = os.path.abspath(os.path.join(args.model_dir_root, prev_task_adapter)) if args.seq_train_type != "multitask" else args.model_dir_root
                model.load_adapter(os.path.abspath(os.path.join(adapter_dir, FINAL_SAVE_NAME + "_adapter")))
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                model.set_active_adapters(adapter_name)
                model.train_adapter(adapter_name)
                
                ### Addition ends here
                
                ## Putting Model on GPU
                model = model.to(args.device_ids[0])

                if not args.fp32:
                    model = FP16_Module(model)
                if args.seq_train_type in REG_TYPE_KEYS:
                    logger.info("calulating reg_params ...")
                    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
                    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
                    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
                    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)
                    regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [train_dataloader], tasks[0])
                    regularizer.task_start_do()
                    regularizer.task_end_do()
                    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))
                    logger.info("done reg_params!")
            args.skip_tasks.remove(tasks[0])
            return model

    model.resize_token_embeddings(len(TOKENIZER))

    if not args.fp32: 
        model = FP16_Module(model)

    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)


    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    if not args.unbound and args.seq_train_type != "multitask":
        n_train_epochs = args.n_train_epochs[tasks[0]]
    else:
        n_train_epochs = args.n_train_epochs['_'.join(tasks)]
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    
    if task_ids[0]==0 and train_type==1:
        # Adding Adapter Modules
        # adapter_config = ConfigUnion(
        #                     PrefixTuningConfig(cross_prefix=True, prefix_length=30, bottleneck_size=800,non_linearity='relu',leave_out=[1,3,5,7,9,11]),
        #                     AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=4, non_linearity="relu", init_weights="mam_adapter", leave_out=[1,3,5,7,9,11]),
        #                 )
        # if args.flat == False:
        #     adapter_config = ConfigUnion(
        #                     PrefixTuningConfig(cross_prefix=True, prefix_length=args.prefixlength, bottleneck_size=args.bottle_neck_size,non_linearity='relu',leave_out=args.leaveout),
        #                     AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=args.rf, non_linearity="relu", init_weights="mam_adapter", leave_out=args.leaveout),
        #                 )
        # else:
        #     adapter_config = ConfigUnion(
        #                     PrefixTuningConfig(cross_prefix=True, prefix_length=args.prefixlength, flat = True,non_linearity='relu',leave_out=args.leaveout),
        #                     AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=args.rf, non_linearity="relu", init_weights="mam_adapter", leave_out=args.leaveout),
        #                 )
        
        adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=args.rf, non_linearity="relu", init_weights="mam_adapter", leave_out=args.leaveout)
        model.add_adapter(tasks[0], config=adapter_config)
        print(model.adapter_summary())
        model.set_active_adapters(tasks[0])
        model.train_adapter(tasks[0])
        model = model.to(args.device_ids[0])
        
    param_optimizer = list(model.named_parameters())  
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if "gem" in args.seq_train_type:
        model.task_id = task_ids[0]
        if not hasattr(model, "grad_dims"):
            model.grad_dims = []
            for param in model.parameters():
                model.grad_dims.append(param.data.numel())
        if not hasattr(model, "grads"):
            model.grads = torch.zeros(sum(model.grad_dims),len(args.tasks))
            model.grads = model.grads.cuda()

    if args.seq_train_type in REG_TYPE_KEYS:
        optimizer = Weight_Regularized_AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.fp32:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                   dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

    scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)),
            num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)
    train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT), args.device_ids)

    if args.seq_train_type in REG_TYPE_KEYS:
        copy_train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
        prev_task = args.tasks[task_ids[0]-1]
        regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [copy_train_dataloader], tasks[0], prev_task)
        regularizer.task_start_do()

    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)
    if "gem" in args.seq_train_type and task_ids[0] != 0:
        gem_step = GEMStep(model, parallel_model, train_loss_fct, optimizer)
    model.train()
    for ep in range(n_train_epochs):
        cum_loss, cum_qa_loss, cum_lm_loss, cur_n_inputs = 0, 0, 0, 0
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y) in enumerate(train_dataloader):

            n_inputs = sum(_cqa.shape[0] for _cqa in cqa)

            for i in range(len(cqa)):
                cqa[i] = (cqa[i].to(args.device_ids[i]),)
                Y[i] = Y[i].to(args.device_ids[i])
                gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                gen_Y[i] = gen_Y[i].to(args.device_ids[i])

            if train_type==1:
                losses = get_losses(parallel_model, cqa, Y, gen_X, gen_Y, train_loss_fct,gen=False)
            else:
                losses = get_losses(parallel_model, cqa, Y, gen_X, gen_Y, train_loss_fct)
            loss = sum(losses)
            if "gem" in args.seq_train_type and task_ids[0] != 0:
                gem_step(task_ids[0])
            train_once(loss, n_inputs)

            qa_loss = losses[0].item() * n_inputs
            lm_loss = losses[1].item() * n_inputs
            cum_loss += (qa_loss + lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cur_n_inputs += n_inputs

            if (n_steps + 1 ) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , avg batch size {:.1f}'.format(
                    ep + cur_n_inputs/len(train_qadata), scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                    cur_n_inputs/(n_steps + 1)
                ))
                
        ### Addition
        if ep==n_train_epochs-1:
            torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME+str(ep+1)))
        if train_type==1:
            adapter_name = args.tasks[task_ids[0]].replace(".", "_")
            model.save_adapter(os.path.join(model_dir, SAVE_NAME+"adapter_"+str(ep+1)), adapter_name)
        tot_n_steps += (n_steps + 1)
        logger.info('epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f} , avg batch size {:.1f}'.format(
            ep+1, n_train_epochs, tot_n_steps, scheduler.get_lr(), cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs, cur_n_inputs/(n_steps+1)
        ))

    if args.seq_train_type in REG_TYPE_KEYS:
        regularizer.task_end_do()
    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))
    if train_type==1:
        print(f"The current active adapter is {model.active_adapters}")
        if task_ids[0]-1 >= 0:
            adapter_name = args.tasks[task_ids[0]].replace(".", "_")
        else:
            adapter_name = args.tasks[task_ids[0]].replace(".", "_")
        print(f"The task with which model is saved {adapter_name}")
        model.save_adapter(os.path.abspath(os.path.join(model_dir, FINAL_SAVE_NAME + "_adapter")), adapter_name)
    return model


if __name__ == '__main__':
    # get the start time
    # set_seed()
    st = time.time()
    st_cpu = time.process_time()
    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    make_dir(args.model_dir_root)

    init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
    logger.info('args = {}'.format(str(args)))
    
    if args.leaveout != "":
        args.leaveout = [int(a) for a in args.leaveout.split(",")]
    else:
        args.leaveout = []

    print(f"\nThe Prefix length is : {args.prefixlength}")
    print(f"The Reduction Facotr is : {args.rf}")
    print(f"The Bottleneck size is : {args.bottle_neck_size}")
    print(f"The leaving layers are : {args.leaveout}")
    print(f"The Flat  : {args.flat}\n")
    
    model = None
    if args.seq_train_type == "multitask":
        model = train(list(range(len(args.tasks))), model)
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound)
        for task_id in range(len(args.tasks)):
            model = train([task_id], model, train_type=1)
    elapsed_time = time.time() - st
    elapsed_time_cpu = time.process_time() - st_cpu
    print('Wall Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print('CPU Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time_cpu)))


#progressive prompts 
#lfpt5