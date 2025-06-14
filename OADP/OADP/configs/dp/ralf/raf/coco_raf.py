_base_ = [
    '../../datasets/ov_coco.py',
    '../../models/oadp_faster_rcnn_r50_fpn.py',
    '../../schedules/40k.py',
    '../../base.py',
]

model = dict(
    global_head=dict(
        classifier=dict(
            type='Classifier',
            prompts='data/prompts/ml_coco.pth',
            out_features=65,
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                prompts='data/prompts/vild.pth',
            ),
        ),
        object_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
        block_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
        test_add_score=dict(
            topk=1,
            is_box_level=True,
            is_with_cc=True, 
            cc_weight_path='ralf/coco_strict_reasoning.pth',  # modify this line to replace with our raf
            concept_pkl_path='ralf/v3det_gpt_noun_chunk_coco_strict_reasoning.pkl',   # modify this line to replace with our noun chunk
        )
    ),
)
trainer = dict(
    optimizer=dict(
        paramwise_cfg=dict(
            custom_keys={
                'roi_head.bbox_head': dict(lr_mult=0.5),
            },
        ),
    ),
    dataloader=dict(
        samples_per_gpu=4, 
        workers_per_gpu=4
    )
)
