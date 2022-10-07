from patch_fitting import * 

if __name__ == "__main__":
    im_name = 'data/akeyboard_small.gif'
    
    place_method = 'entire' # random, entire, subpatch, auto

    Gamma = 0.8

    use_old_cut = True 
    use_grad = False

    im = Image.open(im_name).convert('RGB')
    im_input = np.array(im, dtype=np.uint8)

    h, w, _ = im_input.shape
    height, width = 2 * h, 2 * w
    im_src = np.zeros([height, width, _])
    src_map = np.zeros([height, width]).astype(np.bool)
    seam_map = SeamMap(height, width)


    while not src_map.all():
        region_size = (h // 2, w // 2)
        error_region = get_error_region(im_src, src_map, seam_map, region_size)
        if place_method == 'random':
            offset = get_offset_random(im_src, src_map, im_input)
        elif place_method == 'entire':
            offset = get_offset_entire_matching(im_src, src_map, im_input, None)
        elif place_method == 'subpatch': 
            offset = get_offset_subpatch_matching(im_src, src_map, im_input, error_region, region_size) 
        else: 
            offset = get_offset_auto(im_src, src_map, im_input, error_region, region_size, Gamma)
        print('offset:', offset)
        patch_fitting(im_src, src_map, im_input, offset, seam_map, use_old_cut, use_grad)
        show_im = Image.fromarray(im_src.astype(np.uint8))
        show_im.save('%s-%s.jpg'%(im_name.split('.')[0], place_method))
    