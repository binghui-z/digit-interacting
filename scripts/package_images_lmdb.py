from glob import glob
import sys
sys.path = ['..'] + sys.path
import elytra.sys_utils as sys_utils
#import pdb; pdb.set_trace() # 在可能出错的地方放一个pdb.set_trace()，就可以设置一个断点，直接注释掉

db_path = 'interhand.lmdb'   #.lmdb路径
# fnames = glob('../data/InterHand/images/val/*/*/*/*')

#图片抓取
fnames = glob(r'\\105.1.1.1\Hand\HO&HH\InterHand2.6M\downloads\InterHand2.6M.images.5.fps.v0.0\InterHand2.6M_5fps_batch0\images\val\Capture0\ROM01_No_Interaction_2_Hand\*\*')

# import random
# random.shuffle(fnames)  
# fnames = fnames[:1000]   #把图片随机打乱，并取前1000个。所以为了测试demo,将glob的抓取范围缩小,这一步没必要，暂时注释掉，对所有序列进行测试

map_size = len(fnames) * 5130240     #map的大小为帧数×5130240
# keys = [fname.replace('../data/InterHand/images/', '') for fname in fnames]
keys = [fname.replace(r'\\105.1.1.1\Hand\HO&HH\InterHand2.6M\downloads\InterHand2.6M.images.5.fps.v0.0\InterHand2.6M_5fps_batch0\images', '') for fname in fnames]   #这里应该只是把每一帧的路径截断，然后作为名字

sys_utils.package_lmdb(db_path, map_size, fnames, keys)
