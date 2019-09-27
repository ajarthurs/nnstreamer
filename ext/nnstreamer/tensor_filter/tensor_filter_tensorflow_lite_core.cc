/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file   tensor_filter_tensorflow_lite_core.cc
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @date   7/5/2018
 * @brief  connection with tflite libraries.
 *
 * @bug     No known bugs.
 */

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_conf.h>
#include "tensor_filter_tensorflow_lite_core.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif





#define _print_log(...) if (DBG) g_message (__VA_ARGS__)
/**
 * @brief Compare score of detected objects.
 */
static bool
compare_objs (DetectedObject &a, DetectedObject &b)
{
  return a.prob > b.prob;
}

/**
 * @brief Intersection of union
 */
static gfloat
iou (DetectedObject &A, DetectedObject &B)
{
  int x1 = std::max (A.x, B.x);
  int y1 = std::max (A.y, B.y);
  int x2 = std::min (A.x + A.width, B.x + B.width);
  int y2 = std::min (A.y + A.height, B.y + B.height);
  int w = std::max (0, (x2 - x1 + 1));
  int h = std::max (0, (y2 - y1 + 1));
  float inter = w * h;
  float areaA = A.width * A.height;
  float areaB = B.width * B.height;
  float o = inter / (areaA + areaB - inter);
  return (o >= 0) ? o : 0;
}

/**
 * @brief NMS (non-maximum suppression)
 */
static void
nms (std::vector<DetectedObject> &detected)
{
  const float threshold_iou = 0.5f;
  guint boxes_size;
  guint i, j;

  std::sort (detected.begin (), detected.end (), compare_objs);
  boxes_size = detected.size ();

  std::vector<bool> del (boxes_size, false);
  for (i = 0; i < boxes_size; i++) {
    if (!del[i]) {
      for (j = i + 1; j < boxes_size; j++) {
        if (iou (detected.at (i), detected.at (j)) > threshold_iou) {
          del[j] = true;
        }
      }
    }
  }
  for (i = 0; i < boxes_size; i++) {
    if (del[i]) detected.erase(detected.begin() + i);
  }
  //for (i = 0; i < boxes_size; i++) {
  //  if (!del[i]) {
  //    g_app.detected_objects.push_back (detected[i]);

  //    if (DBG) {
  //      _print_log ("==============================");
  //      _print_log ("Label           : %s",
  //          (gchar *) g_list_nth_data (g_app.tflite_info.labels,
  //              detected[i].class_id));
  //      _print_log ("x               : %d", detected[i].x);
  //      _print_log ("y               : %d", detected[i].y);
  //      _print_log ("width           : %d", detected[i].width);
  //      _print_log ("height          : %d", detected[i].height);
  //      _print_log ("Confidence Score: %f", detected[i].prob);
  //    }
  //  }
  //}
}

#define _expit(x) \
    (1.f / (1.f + expf (-x)))


/**
 * @brief Read strings from file.
 */
static gboolean
read_lines (const gchar * file_name, GList ** lines)
{
  std::ifstream file (file_name);
  if (!file) {
    _print_log ("Failed to open file %s", file_name);
    return FALSE;
  }

  std::string str;
  while (std::getline (file, str)) {
    *lines = g_list_append (*lines, g_strdup (str.c_str ()));
  }

  return TRUE;
}

/**
 * @brief Load box priors.
 */
static gboolean
tflite_load_box_priors (TFLiteModelInfo &tflite_info)
{
  GList *box_priors = NULL;
  gchar *box_row;

  g_return_val_if_fail (read_lines (tflite_info.box_prior_path, &box_priors), FALSE);

  for (int row = 0; row < BOX_SIZE; row++) {
    int column = 0;
    int i = 0, j = 0;
    char buff[11];

    memset (buff, 0, 11);
    box_row = (gchar *) g_list_nth_data (box_priors, row);

    while ((box_row[i] != '\n') && (box_row[i] != '\0')) {
      if (box_row[i] != ' ') {
        buff[j] = box_row[i];
        j++;
      } else {
        if (j != 0) {
          tflite_info.box_priors[row][column++] = atof (buff);
          memset (buff, 0, 11);
        }
        j = 0;
      }
      i++;
    }

    tflite_info.box_priors[row][column++] = atof (buff);
  }

  g_list_free_full (box_priors, g_free);
  return TRUE;
}

/**
 * @brief Load labels.
 */
static gboolean
tflite_load_labels (TFLiteModelInfo &tflite_info)
{
  return read_lines (tflite_info.label_path, &tflite_info.labels);
}

/**
 * @brief Check tflite model and load labels.
 */
static gboolean
tflite_init_info (TFLiteModelInfo &tflite_info, const gchar * path)
{
  tflite_info.label_path = g_strdup_printf ("%s/%s", path, tflite_label);
  tflite_info.box_prior_path =
      g_strdup_printf ("%s/%s", path, tflite_box_priors);

  tflite_info.labels = NULL;

  if (!g_file_test (tflite_info.model_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("the file of model_path is not valid: %s\n", tflite_info.model_path);
    return FALSE;
  }
  if (!g_file_test (tflite_info.label_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("the file of label_path is not valid%s\n", tflite_info.label_path);
    return FALSE;
  }
  if (!g_file_test (tflite_info.box_prior_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("the file of box_prior_path is not valid%s\n", tflite_info.box_prior_path);
    return FALSE;
  }

  g_return_val_if_fail (tflite_load_box_priors (tflite_info), FALSE);
  g_return_val_if_fail (tflite_load_labels (tflite_info), FALSE);

  return TRUE;
}

/**
 * @brief Get detected objects.
 */
static void
get_detected_objects (TFLiteModelInfo &tflite_info, gfloat * detections, gfloat * boxes, std::vector<DetectedObject> &detected)
{
  const float threshold_score = 0.5f;

  for (int d = 0; d < DETECTION_MAX; d++) {
    float ycenter =
        ((boxes[0] / Y_SCALE) * tflite_info.box_priors[2][d]) +
        tflite_info.box_priors[0][d];
    float xcenter =
        ((boxes[1] / X_SCALE) * tflite_info.box_priors[3][d]) +
        tflite_info.box_priors[1][d];
    float h =
        (float) expf (boxes[2] / H_SCALE) * tflite_info.box_priors[2][d];
    float w =
        (float) expf (boxes[3] / W_SCALE) * tflite_info.box_priors[3][d];

    float ymin = ycenter - h / 2.f;
    float xmin = xcenter - w / 2.f;
    float ymax = ycenter + h / 2.f;
    float xmax = xcenter + w / 2.f;

    int x = xmin * MODEL_WIDTH;
    int y = ymin * MODEL_HEIGHT;
    int width = (xmax - xmin) * MODEL_WIDTH;
    int height = (ymax - ymin) * MODEL_HEIGHT;

    for (int c = 1; c < LABEL_SIZE; c++) {
      gfloat score = _expit (detections[c]);
      /**
       * This score cutoff is taken from Tensorflow's demo app.
       * There are quite a lot of nodes to be run to convert it to the useful possibility
       * scores. As a result of that, this cutoff will cause it to lose good detections in
       * some scenarios and generate too much noise in other scenario.
       */
      if (score < threshold_score)
        continue;

      DetectedObject object;

      object.class_id = c;
      object.class_label = (char *) g_list_nth_data (tflite_info.labels, c);
      object.x = x;
      object.y = y;
      object.width = width;
      object.height = height;
      object.prob = score;

      detected.push_back (object);
    }

    detections += LABEL_SIZE;
    boxes += BOX_SIZE;
  }

  nms (detected);
}

///**
// * @brief Callback for tensor sink signal.
// */
//static void
//new_data_cb (GstElement * element, GstBuffer * buffer, gpointer user_data)
//{
//  GstMemory *mem_boxes, *mem_detections;
//  GstMapInfo info_boxes, info_detections;
//  gfloat *boxes, *detections;
//
//  _print_log("tensor_sink called new_data_cb callback");
//  g_return_if_fail (g_app.running);
//
//  /**
//   * tensor type is float32.
//   * [0] dim of boxes > BOX_SIZE : 1 : DETECTION_MAX : 1
//   * [1] dim of labels > LABEL_SIZE : DETECTION_MAX : 1 : 1
//   */
//  g_assert (gst_buffer_n_memory (buffer) == 2);
//
//  /* boxes */
//  mem_boxes = gst_buffer_get_memory (buffer, 0);
//  g_assert (gst_memory_map (mem_boxes, &info_boxes, GST_MAP_READ));
//  g_assert (info_boxes.size == BOX_SIZE * DETECTION_MAX * 4);
//  boxes = (gfloat *) info_boxes.data;
//
//  /* detections */
//  mem_detections = gst_buffer_get_memory (buffer, 1);
//  g_assert (gst_memory_map (mem_detections, &info_detections, GST_MAP_READ));
//  g_assert (info_detections.size == LABEL_SIZE * DETECTION_MAX * 4);
//  detections = (gfloat *) info_detections.data;
//
//  get_detected_objects (detections, boxes);
//
//  gst_memory_unmap (mem_boxes, &info_boxes);
//  gst_memory_unmap (mem_detections, &info_detections);
//
//  gst_memory_unref (mem_boxes);
//  gst_memory_unref (mem_detections);
//}








/**
 * @brief	TFLiteCore creator
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFLiteCore::TFLiteCore (const char * _model_path, nnapi_hw hw)
{
  model_path = _model_path;
  tflite_info.model_path = _model_path;
  if(hw == NNAPI_UNKNOWN){
    use_nnapi = nnsconf_get_custom_value_bool ("tensorflowlite", "enable_nnapi", FALSE);
  } else {
    use_nnapi = TRUE;
  }
  accel = hw;

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
}

/**
 * @brief	TFLiteCore Destructor
 * @return	Nothing
 */
TFLiteCore::~TFLiteCore ()
{
  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
}

/**
 * @brief	initialize the object with tflite model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 */
int
TFLiteCore::init ()
{
  if (!tflite_init_info (tflite_info, "tflite_model")) {
    g_critical ("Failed to initialize TFLite info\n");
    return -4;
  }

  if (loadModel ()) {
    g_critical ("Failed to load model\n");
    return -1;
  }
  if (setInputTensorProp ()) {
    g_critical ("Failed to initialize input tensor\n");
    return -2;
  }
  if (setOutputTensorProp ()) {
    g_critical ("Failed to initialize output tensor\n");
    return -3;
  }
  return 0;
}

/**
 * @brief	get the model path
 * @return the model path.
 */
const char *
TFLiteCore::getModelPath ()
{
  return model_path;
}

/**
 * @brief	load the tflite model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::loadModel ()
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  if (!interpreter) {
    if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
      g_critical ("the file of model_path (%s) is not valid (not regular)\n", model_path);
      return -1;
    }
    model =
        std::unique_ptr <tflite::FlatBufferModel>
        (tflite::FlatBufferModel::BuildFromFile (model_path));
    if (!model) {
      g_critical ("Failed to mmap model\n");
      return -1;
    }
    /* If got any trouble at model, active below code. It'll be help to analyze. */
    /* model->error_reporter (); */

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder (*model, resolver) (&interpreter);
    if (!interpreter) {
      g_critical ("Failed to construct interpreter\n");
      return -2;
    }

    interpreter->UseNNAPI(use_nnapi);

#ifdef ENABLE_NNFW
    if(use_nnapi){
      nnfw_delegate.reset(new ::nnfw::tflite::NNAPIDelegate);
      if(nnfw_delegate->BuildGraph(interpreter.get()) != kTfLiteOk){
	g_critical("Fail to BuildGraph");
	return -3;
      }
    }
#endif
     g_message ("interpreter->UseNNAPI( %s : %s )" , use_nnapi?"true":"false", nnapi_hw_string[accel]);

    /** set allocation type to dynamic for in/out tensors */
    int tensor_idx;

    int tensorSize = interpreter->inputs ().size ();
    for (int i = 0; i < tensorSize; ++i) {
      tensor_idx = interpreter->inputs ()[i];
      interpreter->tensor (tensor_idx)->allocation_type = kTfLiteDynamic;
    }

    tensorSize = interpreter->outputs ().size ();
    for (int i = 0; i < tensorSize; ++i) {
      tensor_idx = interpreter->outputs ()[i];
      interpreter->tensor (tensor_idx)->allocation_type = kTfLiteDynamic;
    }

    if (interpreter->AllocateTensors () != kTfLiteOk) {
      g_critical ("Failed to allocate tensors\n");
      return -2;
    }
  }
#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tfType	: the defined type of Tensorflow Lite
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
TFLiteCore::getTensorType (TfLiteType tfType)
{
  switch (tfType) {
    case kTfLiteFloat32:
      return _NNS_FLOAT32;
    case kTfLiteUInt8:
      return _NNS_UINT8;
    case kTfLiteInt32:
      return _NNS_INT32;
    case kTfLiteBool:
      return _NNS_INT8;
    case kTfLiteInt64:
      return _NNS_INT64;
    case kTfLiteString:
    default:
      /** @todo Support other types */
      break;
  }

  return _NNS_END;
}

/**
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setInputTensorProp ()
{
  auto input_idx_list = interpreter->inputs ();
  inputTensorMeta.num_tensors = input_idx_list.size ();

  for (int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    if (getTensorDim (input_idx_list[i], inputTensorMeta.info[i].dimension)) {
      g_critical ("failed to get the dimension of input tensors");
      return -1;
    }
    inputTensorMeta.info[i].type =
        getTensorType (interpreter->tensor (input_idx_list[i])->type);

#if (DBG)
    gchar *dim_str =
        gst_tensor_get_dimension_string (inputTensorMeta.info[i].dimension);
    g_message ("inputTensorMeta[%d] >> type:%d, dim[%s]",
        i, inputTensorMeta.info[i].type, dim_str);
    g_free (dim_str);
#endif
  }
  return 0;
}

/**
 * @brief extract and store the information of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setOutputTensorProp ()
{
  auto output_idx_list = interpreter->outputs ();
  outputTensorMeta.num_tensors = output_idx_list.size ();

  for (int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    if (getTensorDim (output_idx_list[i], outputTensorMeta.info[i].dimension)) {
      g_critical ("failed to get the dimension of output tensors");
      return -1;
    }
    outputTensorMeta.info[i].type =
        getTensorType (interpreter->tensor (output_idx_list[i])->type);

#if (DBG)
    gchar *dim_str =
        gst_tensor_get_dimension_string (outputTensorMeta.info[i].dimension);
    g_message ("outputTensorMeta[%d] >> type:%d, dim[%s]",
        i, outputTensorMeta.info[i].type, dim_str);
    g_free (dim_str);
#endif
  }
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param tensor_idx	: the real index of model of the tensor
 * @param[out] dim	: the array of the tensor
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getTensorDim (int tensor_idx, tensor_dim dim)
{
  int len = interpreter->tensor (tensor_idx)->dims->size;
  g_assert (len <= NNS_TENSOR_RANK_LIMIT);

  /* the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (interpreter->tensor (tensor_idx)->dims->data,
      interpreter->tensor (tensor_idx)->dims->data + len, dim);

  /* fill the remnants with 1 */
  for (int i = len; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 1;
  }

  return 0;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getInputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &inputTensorMeta);
  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getOutputTensorDim (GstTensorsInfo * info)
{
  gst_tensors_info_copy (info, &outputTensorMeta);
  return 0;
}

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::invoke (const GstTensorMemory * input, GstTensorMemory * output)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  std::vector <int> tensors_idx;
  int tensor_idx;
  TfLiteTensor *tensor_ptr;

  for (int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->outputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == output[i].size);
    tensor_ptr->data.f = (gfloat *) output[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  for (int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->inputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == input[i].size);
    tensor_ptr->data.f = (gfloat *) input[i].data;
    tensors_idx.push_back (tensor_idx);
  }

#ifdef ENABLE_NNFW
  if(use_nnapi){
    if(nnfw_delegate->Invoke(interpreter.get()) != kTfLiteOk){
      g_critical ("Failed to invoke");
      return -3;
    }
  }else{
    if (interpreter->Invoke () != kTfLiteOk) {
      g_critical ("Failed to invoke");
      return -3;
    }
  }
#else
  if (interpreter->Invoke () != kTfLiteOk) {
    g_critical ("Failed to invoke");
    return -3;
  }
#endif

  /** if it is not `nullptr`, tensorflow makes `free()` the memory itself. */
  int tensorSize = tensors_idx.size ();
  for (int i = 0; i < tensorSize; ++i) {
    interpreter->tensor (tensors_idx[i])->data.raw = nullptr;
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  return 0;
}

/**
 * @brief	call the creator of TFLiteCore class.
 * @param	_model_path	: the logical path to '{model_name}.tffile' file
 * @return	TFLiteCore class
 */
void *
tflite_core_new (const char * _model_path, nnapi_hw hw)
{
  return new TFLiteCore (_model_path, hw);
}

/**
 * @brief	delete the TFLiteCore class.
 * @param	tflite	: the class object
 * @return	Nothing
 */
void
tflite_core_delete (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  delete c;
}

/**
 * @brief	initialize the object with tflite model
 * @param	tflite	: the class object
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_init (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->init ();
}

/**
 * @brief	get the model path
 * @param	tflite	: the class object
 * @return the model path.
 */
const char *
tflite_core_getModelPath (void * tflite)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getModelPath ();
}

/**
 * @brief	get the Dimension of Input Tensor of model
 * @param	tflite	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getInputDim (void * tflite, GstTensorsInfo * info)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getInputTensorDim (info);
}

/**
 * @brief	get the Dimension of Output Tensor of model
 * @param	tflite	: the class object
 * @param[out] info Structure for tensor info.
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_getOutputDim (void * tflite, GstTensorsInfo * info)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->getOutputTensorDim (info);
}

/**
 * @brief	invoke the model
 * @param	tflite	: the class object
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
tflite_core_invoke (void * tflite, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  TFLiteCore *c = (TFLiteCore *) tflite;
  return c->invoke (input, output);
}
