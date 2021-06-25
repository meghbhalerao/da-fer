Log_Name='raf2aisin_res50'
##Resume_Model='ResNet50_CropNet_withoutAFN_trainOnSourceDomain_RAFtoCK+.pkl'
Resume_Model=None
OutputPath='.'
GPU_ID=1
Backbone='ResNet50'
useAFN='False'
methodOfAFN='SAFN'
radius=25
deltaRadius=1
weight_L2norm=0.05
useDAN='True'
methodOfDAN='MME'
faceScale=112
sourceDataset='RAF'
targetDataset='AISIN'
train_batch_size=32
test_batch_size=32
useMultiDatasets='False'
epochs=100
lr=0.0001
lr_ad=0.001
momentum=0.9
weight_decay=0.0001
isTest='False'
showFeature='False'
class_num=2
useIntraGCN='True'
useInterGCN='True'
useLocalFeature='True'
useRandomMatrix='False'
useAllOneMatrix='False'
useCov='False'
useCluster='False'

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python3 TransferToTargetDomain.py \
--log ${Log_Name} \
--out ${OutputPath} \
--net ${Backbone} \
--pretrained ${Resume_Model} \
--dev ${GPU_ID} \
--use_afn ${useAFN} \
--afn_method ${methodOfAFN} \
--r ${radius} \
--dr ${deltaRadius} \
--w_l2 ${weight_L2norm} \
--use_dan ${useDAN} \
--dan_method ${methodOfDAN} \
--face_scale ${faceScale} \
--source ${sourceDataset} \
--target ${targetDataset} \
--train_batch ${train_batch_size} \
--test_batch ${test_batch_size} \
--multiple_data ${useMultiDatasets} \
--epochs ${epochs} \
--lr ${lr} \
--lr_ad ${lr_ad} \
--momentum ${momentum} \
--weight_decay ${weight_decay} \
--isTest ${isTest} \
--show_feat ${showFeature} \
--class_num ${class_num} \
--intra_gcn ${useIntraGCN} \
--inter_gcn ${useInterGCN} \
--local_feat ${useLocalFeature} \
--rand_mat ${useRandomMatrix} \
--all1_mat ${useAllOneMatrix} \
--use_cov ${useCov} \
--use_cluster ${useCluster}
