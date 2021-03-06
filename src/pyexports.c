#include "darknet.h"
#include "stb_image.h"

extern image load_image_from_memory_thread(stbi_uc const *data,
                                           int len,
                                           int w,
                                           int h,
                                           int c,
                                           pthreadpool_t threadpool);

int create_yolo_handle(void **net, char ***names, const char *name_list,
                       const char *cfgfile, const char *weightfile,
                       int init_nnp)
{
#ifndef NNPACK
    return -1;
#else
    network *netp = load_network((char *)cfgfile, (char *)weightfile, 0);
    if (init_nnp) {
        nnp_initialize();
    }
    *net = (void *)netp;
    *names = get_labels((char *)name_list);
    if (*names == NULL) {
        return -1;
    }
    return 0;
#endif

}

int detect_image(void *p, unsigned char *data, int len, char **names,
                 int num_threads,
                 float thresh, float hier_thresh)
{
#ifndef NNPACK
    return -1;
#else
    network *net = p;
    box *boxes;
    layer l;
    float **probs = NULL;
    float **masks = NULL;
    int i, j;
    int status = 0;

    net->threadpool = pthreadpool_create(num_threads);
    image im = load_image_from_memory_thread(data, len, 0, 0, net->c, net->threadpool);
    image sized = letterbox_image_thread(im, net->w, net->h, net->threadpool);

    l = net->layers[net->n-1];
    /* if boxes and probs are not allocated, we allocate them here */
    boxes = calloc(l.w*l.h*l.n, sizeof(box));
    if (boxes == NULL){
        status = -1;
        goto err;
    }
    probs = calloc(l.w*l.h*l.n, sizeof(float *));
    if (!probs) {
        status = -1;
        goto errfreeboxes;
    }
    for(j = 0; j < l.w*l.h*l.n; ++j){
        probs[j] = calloc(l.classes + 1, sizeof(float));
        if (!probs[j]) {
            status = -1;
            goto errfreeprobs;
        }
    }
    if (l.coords > 4){
        masks = calloc(l.w*l.h*l.n, sizeof(float*));
        if (!masks) {
            status = -1;
            goto errfreeprobs;
        }
        for(j = 0; j < l.w*l.h*l.n; ++j){
            masks[j] = calloc(l.coords-4, sizeof(float));
            if (!masks[j]) {
                status = -1;
                goto errfreemasks;
            }
        }
    }

    float *X = sized.data;
    network_predict(net, X);
    get_region_boxes(l, im.w, im.h, net->w, net->h,
                     thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);

    for (i = 0; i < l.w*l.h*l.n; ++i) {
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < l.classes; ++j){
            if (probs[i][j] > thresh){
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], probs[i][j]*100);
            }
        }
    }
    draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, NULL, l.classes);
    free_image(im);
    free_image(sized);

errfreemasks:
    if (l.coords > 4){
        for(i = 0; i < l.w*l.h*l.n; ++i){
            if (masks[i]){
                free(masks[i]);
            }
        }
        free(masks);
    }
errfreeprobs:
    for(i = 0; i < l.w*l.h*l.n; ++i){
        free(probs[i]);
    }
    free(probs);
errfreeboxes:
    free(boxes);
err:
    pthreadpool_destroy(net->threadpool);
    return status;
#endif
}

int free_yolo_network_handle(void *p, int deinit_nnp)
{
#ifndef NNPACK
    return -1;
#else
    if (deinit_nnp) {
        nnp_deinitialize();
    }
    free_network(p);
    return 0;
#endif
}

/*
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
	struct timeval start, stop;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.3;
#ifdef NNPACK
	nnp_initialize();
	net->threadpool = pthreadpool_create(4);
#endif

    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
		}
#ifdef NNPACK
		image im = load_image_thread(input, 0, 0, net->c, net->threadpool);
		image sized = letterbox_image_thread(im, net->w, net->h, net->threadpool);
#else
		image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
#endif
        layer l = net->layers[net->n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
        float **masks = 0;
        if (l.coords > 4){
            masks = calloc(l.w*l.h*l.n, sizeof(float*));
            for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = calloc(l.coords-4, sizeof(float *));
        }

        float *X = sized.data;
		gettimeofday(&start, 0);
		network_predict(net, X);
		gettimeofday(&stop, 0);
		printf("%s: Predicted in %ld ms.\n", input, (stop.tv_sec * 1000 + stop.tv_usec / 1000) - (start.tv_sec * 1000 + start.tv_usec / 1000));
        get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
            cvWaitKey(0);
            cvDestroyAllWindows();
#endif
        }

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        if (filename) break;
    }
#ifdef NNPACK
	pthreadpool_destroy(net->threadpool);
	nnp_deinitialize();
#endif
}
*/
