Log_Name='raf2aisin_res50'
Resume_Model=None
OutputPath='.'
GPU_ID=0    
Backbone='ResNet50'
useAFN='False'
methodOfAFN='SAFN'
radius=40
deltaRadius=0.001
weight_L2norm=0.05
faceScale=112
sourceDataset='RAF'
targetDataset='AISIN'
train_batch_size=32
test_batch_size=32
useMultiDatasets='False'
epochs=60
lr=0.0001
momentum=0.9
weight_decay=0.0001
isTest='False'
showFeature='False'
class_num=2
useIntraGCN='False'
useInterGCN='False'
useLocalFeature='True'
useRandomMatrix='False'
useAllOneMatrix='False'
useCov='False'
useCluster='False'
     
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python3 TrainOnSourceDomain.py \
--Log_Name ${Log_Name} \
--OutputPath ${OutputPath} \
--pretrained ${Resume_Model} \
--GPU_ID ${GPU_ID} \
--net ${Backbone} \
--useAFN ${useAFN} \
--methodOfAFN ${methodOfAFN} \
--radius ${radius} \
--deltaRadius ${deltaRadius} \
--weight_L2norm ${weight_L2norm} \
--face_scale ${faceScale} \
--source ${sourceDataset} \
--target ${targetDataset} \
--train_batch ${train_batch_size} \
--test_batch ${test_batch_size} \
--useMultiDatasets ${useMultiDatasets} \
--epochs ${epochs} \
--lr ${lr} \
--momentum ${momentum} \
--weight_decay ${weight_decay} \
--isTest ${isTest} \
--showFeature ${showFeature} \
--class_num ${class_num} \
--intra_gcn ${useIntraGCN} \
--inter_gcn ${useInterGCN} \
--local_feat ${useLocalFeature} \
--rand_mat ${useRandomMatrix} \
--all1_mat ${useAllOneMatrix} \
--use_cov ${useCov} \
--use_cluster ${useCluster}