"""
Stitching sample (advanced)
===========================

Show how to use Stitcher API from python.


use case 
===========================
Couple of words about the use case:
 Stitching algorithm is designed with assumption that images contains enough texture details (features)
  and objects on the frames are far from camera (distance to objects much much larger than camera translation).
   The second item is not true for the provided images. It means that, most probably, it's not bundle adjustment bug.
    We just should ensure that all corner cases are handled correctly
"""

# Python 2/3 compatibility
from __future__ import print_function

import argparse
from collections import OrderedDict

import cv2 as cv
import numpy as np

from os import listdir
from os.path import isfile, join

EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster

FEATURES_FIND_CHOICES = OrderedDict()
try:
    cv.xfeatures2d_SURF.create() # check if the function can be called
    FEATURES_FIND_CHOICES['surf'] = cv.xfeatures2d_SURF.create
except (AttributeError, cv.error) as e:
    print("SURF not available")
# if SURF not available, ORB is default
FEATURES_FIND_CHOICES['orb'] = cv.ORB.create
try:
    FEATURES_FIND_CHOICES['sift'] = cv.SIFT_create
except AttributeError:
    print("SIFT not available")
try:
    FEATURES_FIND_CHOICES['brisk'] = cv.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['akaze'] = cv.AKAZE_create
except AttributeError:
    print("AKAZE not available")

SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
SEAM_FIND_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
SEAM_FIND_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator
ESTIMATOR_CHOICES['affine'] = cv.detail_AffineBasedEstimator

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

WAVE_CORRECT_CHOICES = OrderedDict()
WAVE_CORRECT_CHOICES['horiz'] = cv.detail.WAVE_CORRECT_HORIZ
WAVE_CORRECT_CHOICES['no'] = None
WAVE_CORRECT_CHOICES['vert'] = cv.detail.WAVE_CORRECT_VERT

BLEND_CHOICES = ('multiband', 'feather', 'no',)

# match_conf = 0.65

parser = argparse.ArgumentParser(
    prog="stitching_detailed.py", description="Rotation model images stitcher"
)
parser.add_argument(
    'img_names', nargs='+',
    help="Files to stitch", type=str
)
parser.add_argument(
    '--try_cuda',
    action='store',
    default=False,
    help="Try to use CUDA. The default value is no. All default values are for CPU mode.",
    type=bool, dest='try_cuda'
)
parser.add_argument(
    '--work_megapix', action='store', default=0.6,
    help="Resolution for image registration step. The default is 0.6 Mpx",
    type=float, dest='work_megapix'
)
parser.add_argument(
    '--features', action='store', default=list(FEATURES_FIND_CHOICES.keys())[0],
    help="Type of features used for images matching. The default is '%s'." % list(FEATURES_FIND_CHOICES.keys())[0],
    choices=FEATURES_FIND_CHOICES.keys(),
    type=str, dest='features'
)
parser.add_argument(
    '--matcher', action='store', default='homography',
    help="Matcher used for pairwise image matching. The default is 'homography'.",
    choices=('homography', 'affine'),
    type=str, dest='matcher'
)
parser.add_argument(
    '--estimator', action='store', default=list(ESTIMATOR_CHOICES.keys())[0],
    help="Type of estimator used for transformation estimation. The default is '%s'." % list(ESTIMATOR_CHOICES.keys())[0],
    choices=ESTIMATOR_CHOICES.keys(),
    type=str, dest='estimator'
)
parser.add_argument(
    '--match_conf', action='store',
    help="Confidence for feature matching step. The default is 0.3 for ORB and 0.65 for other feature types, Match distances ration threshold",
    type=float, dest='match_conf'
)
parser.add_argument(
    '--conf_thresh', action='store', default=1.0,
    help="Threshold for two images are from the same panorama confidence.The default is 1.0",
    type=float, dest='conf_thresh'
)
parser.add_argument(
    '--ba', action='store', default=list(BA_COST_CHOICES.keys())[0],
    help="Bundle adjustment cost function. The default is '%s'." % list(BA_COST_CHOICES.keys())[0],
    choices=BA_COST_CHOICES.keys(),
    type=str, dest='ba'
)
parser.add_argument(
    '--ba_refine_mask', action='store', default='xxxxx',
    help="Set refinement mask for bundle adjustment. It looks like 'x_xxx', "
         "where 'x' means refine respective parameter and '_' means don't refine, "
         "and has the following format:<fx><skew><ppx><aspect><ppy>. "
         "The default mask is 'xxxxx'. "
         "If bundle adjustment doesn't support estimation of selected parameter then "
         "the respective flag is ignored.",
    type=str, dest='ba_refine_mask'
)
parser.add_argument(
    '--wave_correct', action='store', default=list(WAVE_CORRECT_CHOICES.keys())[0],
    help="Perform wave effect correction. The default is '%s'" % list(WAVE_CORRECT_CHOICES.keys())[0],# this determin the angle of camera center
    choices=WAVE_CORRECT_CHOICES.keys(),
    type=str, dest='wave_correct'
)
parser.add_argument(
    '--save_graph', action='store', default=None,
    help="Save matches graph represented in DOT language to <file_name> file.",
    type=str, dest='save_graph'
)
parser.add_argument(
    '--warp', action='store', default=WARP_CHOICES[0],
    help="Warp surface type. The default is '%s'." % WARP_CHOICES[0],
    choices=WARP_CHOICES,
    type=str, dest='warp'
)
parser.add_argument(
    '--seam_megapix', action='store', default=0.1,
    help="Resolution for seam estimation step. The default is 0.1 Mpx.",
    type=float, dest='seam_megapix'
)
parser.add_argument(
    '--seam', action='store', default=list(SEAM_FIND_CHOICES.keys())[0],
    help="Seam estimation method. The default is '%s'." % list(SEAM_FIND_CHOICES.keys())[0],
    choices=SEAM_FIND_CHOICES.keys(),
    type=str, dest='seam'
)
parser.add_argument(
    '--compose_megapix', action='store', default=5,
    help="Resolution for compositing step. Use -1 for original resolution. The default is -1",
    type=float, dest='compose_megapix'
)
parser.add_argument(
    '--expos_comp', action='store', default=list(EXPOS_COMP_CHOICES.keys())[0],
    help="Exposure compensation method. The default is '%s'." % list(EXPOS_COMP_CHOICES.keys())[0],
    choices=EXPOS_COMP_CHOICES.keys(),
    type=str, dest='expos_comp'
)
parser.add_argument(
    '--expos_comp_nr_feeds', action='store', default=1,
    help="Number of exposure compensation feed.",
    type=np.int32, dest='expos_comp_nr_feeds'
)
parser.add_argument(
    '--expos_comp_nr_filtering', action='store', default=2,
    help="Number of filtering iterations of the exposure compensation gains.",
    type=float, dest='expos_comp_nr_filtering'
)
parser.add_argument(
    '--expos_comp_block_size', action='store', default=32,
    help="BLock size in pixels used by the exposure compensator. The default is 32.",
    type=np.int32, dest='expos_comp_block_size'
)
parser.add_argument(
    '--blend', action='store', default=BLEND_CHOICES[0],
    help="Blending method. The default is '%s'." % BLEND_CHOICES[0],
    choices=BLEND_CHOICES,
    type=str, dest='blend'
)
parser.add_argument(
    '--blend_strength', action='store', default=5,
    help="Blending strength from [0,100] range. The default is 5",
    type=np.int32, dest='blend_strength'
)
parser.add_argument(
    '--output', action='store', default='result.jpg',
    help="The default is 'result.jpg'",
    type=str, dest='output'
)
parser.add_argument(
    '--timelapse', action='store', default=None,#  None, 'as_is', 'crop'
    help="Output warped images separately as frames of a time lapse movie, "
         "with 'fixed_' prepended to input file names.",
    type=str, dest='timelapse'
)
parser.add_argument(
    '--rangewidth', action='store', default=-1,
    help="uses range_width to limit number of images to match with.",
    type=int, dest='rangewidth'
)
parser.add_argument(
    '--show_match_preview', action='store', default=False,
    help="assign True is you want to preview mathes result",
    type=bool, dest='show_match_preview'
)

parser.add_argument(
    '--show_keypoint_preview', action='store', default=False,
    help="assign True is you want to preview find features result",
    type=bool, dest='show_keypoint_preview'
)

__doc__ += '\n' + parser.format_help()


def get_matcher(args):
    try_cuda = args.try_cuda
    matcher_type = args.matcher
    if args.match_conf is None:
        if args.features == 'orb':
            match_conf = 0.3
        else:
            match_conf = 0.65
    else:
        match_conf = args.match_conf
    print("match confidence : " + str(match_conf))
    range_width = args.rangewidth
    if matcher_type == "affine":
        matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
    elif range_width == -1:
        matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
    else:
        matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda, match_conf)
    return matcher


def get_compensator(args):
    expos_comp_type = EXPOS_COMP_CHOICES[args.expos_comp]
    expos_comp_nr_feeds = args.expos_comp_nr_feeds
    expos_comp_block_size = args.expos_comp_block_size
    # expos_comp_nr_filtering = args.expos_comp_nr_filtering
    if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
        compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator


def main():
    path = "./use_image/"
    # path = "./compose_low_high_image/"
    onlyImages = [path + f for f in listdir(path) if isfile(join(path,f))]
   
    args = parser.parse_args()
    args.img_names = onlyImages
    img_names = args.img_names
    print(img_names)
    work_megapix = args.work_megapix
    seam_megapix = args.seam_megapix
    compose_megapix = args.compose_megapix
    conf_thresh = args.conf_thresh
    match_conf = args.match_conf
    ba_refine_mask = args.ba_refine_mask
    wave_correct = WAVE_CORRECT_CHOICES[args.wave_correct]
    if args.save_graph is None:
        save_graph = False
    else:
        save_graph = True
    warp_type = args.warp
    blend_type = args.blend
    blend_strength = args.blend_strength
    result_name = args.output
    if args.timelapse is not None:
        timelapse = True
        if args.timelapse == "as_is":
            timelapse_type = cv.detail.Timelapser_AS_IS
        elif args.timelapse == "crop":
            timelapse_type = cv.detail.Timelapser_CROP
        else:
            print("Bad timelapse method")
            exit()
    else:
        timelapse = False
    finder = FEATURES_FIND_CHOICES[args.features]()
    print("use finder : " + args.features)
    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    full_size_img_set = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False

    ######################## Find features #######################################
    print("-- Find Features --")
    for name in img_names:
        full_img = cv.imread(cv.samples.findFile(name))
        if full_img is None:
            print("Cannot read image ", name)
            continue
        
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        
        convert_grayImg = True
        if convert_grayImg :
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_feat = cv.detail.computeImageFeatures2(finder, gray_img)
        else:
            img_feat = cv.detail.computeImageFeatures2(finder, img)
        
        ###################### Draw Features and key point #######################
        show_keypoint = args.show_keypoint_preview
        if show_keypoint:
            keypoint_img = cv.drawKeypoints(img, img_feat.getKeypoints(), None, (255, 0, 0), 4)
            keypoint_img = cv.resize(src=keypoint_img, dsize=(1024, 768), interpolation=cv.INTER_LINEAR_EXACT)
            cv.imshow("KeyPoint" , keypoint_img)
            cv.waitKey()
        ######################################################################
        features.append(img_feat)
        full_size_img_set.append(img)
        img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        images.append(img)
    
    cv.destroyAllWindows()
       
    ###################### Match Feature #####################################
    print("-- Match Feature --")
    # find homography has been call internally (inside the BestOf2NearestMatcher which is the matcher)
    matcher = get_matcher(args)
    p = matcher.apply2(features)

        
    # Draw Matches
    Draw_Matches = args.show_match_preview
    if Draw_Matches:
        for pair in p :
            # print(dir(pair))
            if pair.confidence < match_conf:
                continue
            matches = pair.getMatches()
            src_img = full_size_img_set[pair.src_img_idx]
            dst_img = full_size_img_set[pair.dst_img_idx]
            src_kp = features[pair.src_img_idx].getKeypoints()
            dst_kp = features[pair.dst_img_idx].getKeypoints()
            draw_parms = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = None, flags = 2)
            matches_img = cv.drawMatches(src_img, src_kp, dst_img, dst_kp, matches, None, **draw_parms)
            matches_img = cv.resize(matches_img, (1024, 764), interpolation=cv.INTER_LINEAR_EXACT)
            cv.imshow("match img",matches_img)
            cv.waitKey()
    
    full_size_img_set.clear()
    matcher.collectGarbage()

    # Record matches graph everytime
    if True:
        with open('./matchgraph.txt', 'w') as fh:
            fh.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))

    # find a certain image of pano
    indices = cv.detail.leaveBiggestComponent(features, p, conf_thresh)
    ###################### Probabilistic Model for Image Match Verification #####
    img_subset = []
    img_names_subset = []
    full_img_sizes_subset = []
    features_subset = []
    for i in range(len(indices)):
        print(img_names[indices[i, 0]])
        img_names_subset.append(img_names[indices[i, 0]])
        img_subset.append(images[indices[i, 0]])
        full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
        features_subset.append(features[indices[i, 0]])
    # Remove unsuitable feature and pairwise_matches (test)
    features = features_subset
    p = matcher.apply2(features)
    matcher.collectGarbage()

    images = img_subset
    img_names = img_names_subset
    full_img_sizes = full_img_sizes_subset
    num_images = len(img_names)
    print(num_images)
    if num_images < 2:
        print("Need more images")
        exit()
        

    ###################### Estimated Cameras Parameter ######################
    print("-- Estimated Cameras Parameter --")
    estimator = ESTIMATOR_CHOICES[args.estimator]()
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        exit()

    #cam R is the rotation matrix which is the special case of homography matrix that require the focal length
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
        # if cam.focal < 0:
        #     cam.focal = np.sqrt(cam.focal ** 2)
        #     print('modify camera focal length to positive')
    
    ###################### Bundle Adjustiment ########################
    print("-- Bundle Adjustiment --")
    adjuster = BA_COST_CHOICES[args.ba]()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        exit()
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    
    ###################### Automatic Panorama Straightening ########## 釐清
    print("-- Automatic Panorama Straightening --")
    if wave_correct is not None:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
        # 設定環景圖旋轉角度
        rmats = cv.detail.waveCorrect(rmats, wave_correct)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]

    ###################### Warp Image to spherical space #################################
    print("-- Warp Image to spherical space --")
    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)
    
    
    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    show_warp_img = False
    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)
        if show_warp_img:
            cv.imshow("warp_img", img)
            cv.waitKey()
        cv.destroyAllWindows()
    ########################### Gain Compensation #################################
    print("-- Gain Compensation --")
    compensator = get_compensator(args)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    seam_finder = SEAM_FIND_CHOICES[args.seam]
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)

    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    timelapser = None
    # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
    for idx, name in enumerate(img_names):
        full_img = cv.imread(name)
        # MAX_VALUE = 1020
        # full_img = cv.resize(img, (MAX_VALUE, int(MAX_VALUE/img.shape[1] * img.shape[0])), interpolation=cv.INTER_LINEAR)
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            # warped image scale : the median of cameras focals
            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
            # recalculate all cameras parameter
            for i in range(0, len(img_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (int(full_img_sizes[i][0] * compose_scale), int(full_img_sizes[i][1] * compose_scale))
                # print(sz)
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])

        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img
        _img_size = (img.shape[1], img.shape[0])

        
        ###################### Blending################################################
        print("-- Blending --")
        print(" image size : " + str(_img_size))
        # img = cv.resize(img, (MAX_VALUE, int(MAX_VALUE/img.shape[1] * img.shape[0])), interpolation=cv.INTER_LINEAR)
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv.dilate(masks_warped[idx], None)
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)
        if blender is None and not timelapse:
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(int))
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        elif timelapser is None and timelapse:
            timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
            timelapser.initialize(corners, sizes)
        if timelapse:
            ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
            timelapser.process(image_warped_s, ma_tones, corners[idx])
            pos_s = img_names[idx].rfind("/")
            if pos_s == -1:
                fixed_file_name = "fixed_" + img_names[idx]
            else:
                fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
            cv.imwrite(fixed_file_name, timelapser.getDst())
        else:
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
    if not timelapse:
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        cv.imwrite(result_name, result)
        zoom_x = 600.0 / result.shape[1]
        dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        cv.imshow(result_name, dst)
        cv.waitKey()

    print("Done")


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
