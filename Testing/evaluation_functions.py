# Jaccard similarity for checking the performance of the model

def jaccard(tp, fp, fn):
  den = fp+tp+fn

  if den==0:
    return 1
  else:
    return round(tp/den, 2)


# Dice Score for checking the performance of the model

def dice_score(tp, fp, fn):
  den = fp+2*tp+fn

  if den==0:
    return 1
  else:
    return round(2*tp/den, 2)


# Precision for checking the performance of the model

def precision(tp, fp):
  den = tp+fp

  if den==0:
    return 1
  else:
    return round(tp/den, 2)


# Sensitivity measure for checking the performance of the model

def sensitivity(tp, fn):      # same as recall
  den = tp+fn

  if den==0:
    return 1
  else:
    return round(tp/den, 2)


# Accuracy for checking the performance of the model

def accuracy(tp, total):
  total_tp = np.sum(tp)
  return round(total_tp/total, 2)


'''
Function for getting the class in which each pixel of the output belongs, and is stored in 'output' varible.

The model returns a single, floating point value between 0 and 1 for each pixel.
The segmentation task was treated as a classication task for each pixel and there are 2 classes in total.
Values close to 0 is considered to be in class 0, otherwise in class 1

The function then calculates the true positive, false positive and false negative count for class 1,
which is the class denoting the presence of a solar panel.

Two variables can be passed to the function:
The 'output' variable has the model's output and the 'actual' variable has the ground truth.

'actual' has a default value of [] to indicate that the updated 'output' variable is to be returned,
otherwise the tp, fp, fn counts are returned.

'ds' parameter checks for deep supervision
'''

def get_count(output, actual=[], ds=0):

  tp = np.zeros(2)
  fp = np.zeros(2)
  fn = np.zeros(2)

  output = np.where(output<0.5, 0, 1)
  output = np.array(output, dtype=np.uint8)

  s = output.shape

  if not list(actual):
    return output

  for i in range(s[0]):
    for j in range(s[1]):

      if output[i, j]==actual[i, j]:
        tp[int(output[i, j])] += 1
        # rat[int(actual[i, j])] += 1

      elif (not output[i, j]==actual[i, j]):
        fp[int(output[i, j])] += 1
        fn[int(actual[i, j])] += 1

        # rat[int(actual[i, j])] += 1
        # rat[int(output[i, j])] += 1

  return tp, fp, fn


# Function for calculating the evaluation metrics for each class
# (using the true positive, false positive and false negative count for each class)

# For normal images, the average of the metrics calculated for both the classes is returned (class 0: background, class 1: solar panels)
# For images having no solar panels, the metrics are completely calculated on the background pixels
# 'avg' variable determines which of the above two operations is to be performed


def calc_met(tp, fp, fn, total, avg):

  if avg==False:

    tp = sum(tp)
    fp = sum(fp)
    fn = sum(fn)

    avg_prec = precision(tp, fp)
    avg_sens = sensitivity(tp, fn)
    avg_jac = jaccard(tp, fp, fn)
    avg_dice = dice_score(tp, fp, fn)
    acc = round(accuracy(tp, total), 2)


  elif avg==True:

    avg_prec = precision(tp[1], fp[1])
    avg_sens = sensitivity(tp[1], fn[1])
    avg_jac = jaccard(tp[1], fp[1], fn[1])
    avg_dice = dice_score(tp[1], fp[1], fn[1])
    acc = round(accuracy(sum(tp), total), 2)

  return avg_prec, avg_sens, avg_jac, avg_dice, acc


# Function for printing the mean and the standard deviation of the evaluation metric for segmentation over all the images

def cal_avg_metric(metrics):

  c = 1
  for i in metrics:

    i_mean = round(np.mean(i), 2)
    i_std = round(np.std(i), 2)

    print('\nMetrics no.', c)
    print(i_mean, i_std)
    c += 1


'''
Function for prediction and evaluation for the segmentation task

For each test image, the segmentation is done then passed on to the two functions:
'get_count' and 'calc_met' for finally getting the evaluation metric values per image.

This process is repeated for all the test images to be evaluated.

The values returned by calc_met are added to a list for getting the mean and SD of those values over all the test images
with the 'cal_avg_metric' function

This function can also return the segmentation, instead of the metric values, in which case,
nothing should be passed to the variables 'y_test' so they are initialised to their default values

The variable 'per' determines what percent of the images not containing any solar panel to be included for the testing purposes

'ds' parameter checks for deep supervision
'''

def pred_and_eval(model, X_test, y_test=[], per=0, p=1, ds=0):

  if len(X_test.shape)<4:       # For a single test case, converting the 3D slice to a 4D array
    X_test = np.array([X_test])

    if list(y_test):
      y_test = np.array([y_test])

  prec_list = []
  dice_list = []
  jac_list = []
  sens_list = []
  acc_list = []


  new_out = []
  num = len(X_test)       # number of test cases
  output = model.predict(X_test)
  output = np.array(output)
  sh = output.shape

  if not list(y_test):          # In case, y_test is empty, return the model output after proper post-processing
    output = get_count(output, ds=ds)
    output = np.reshape(output*255, (-1, sh[1], sh[2]))
    return output

  for k in range(num):
    actual = y_test[k]
    avg = True

    if not 1 in actual:     # Checking whether the ground truth contains any solar panel or not
      f = np.random.randint(1, 101)

      if f>=per:
        continue
      else:
        avg = False

    if ds==0:
      tp, fp, fn = get_count(output[k], actual)       # Calculate the tp, fp, fn of the test images
    else:
      tp, fp, fn = get_count(output[:, k], actual, 1)

    prec, sens, jac, dice, acc = calc_met(tp, fp, fn, sh[1]*sh[2], avg)    # Calculating the metrics

    acc_list.append(acc)
    prec_list.append(prec)
    dice_list.append(dice)
    jac_list.append(jac)
    sens_list.append(sens)

  if num==1:                    # For num = 1, the metric values will be displayed directly,
                                # instead of the mean and SD, since they store only a single value
    if dice_list and (p==1 or dice_list[0] < 0.9):    # For checking the samples which have a low dice score

      # print('Metric values are: ')

      # print('Precision: ', prec_list[0])
      # print('Recall: ', sens_list[0])
      # print('Jaccard Similarity: ', jac_list[0])
      # print('Dice Score: ', dice_list[0])
      # print('Accuracy: ', acc_list[0])

      # if p==0 and dice_list[0] < 0.9:
      # if avg==False:

        # print('Metric values are: ')

      # print('Precision: ', prec_list[0])
      # print('Recall: ', sens_list[0])
      # print('Jaccard Similarity: ', jac_list[0])
      # print('Dice Score: ', dice_list[0])
      # print('Accuracy: ', acc_list[0])

      cv2_imshow(np.reshape(X_test*128, (sh[1], sh[2], 3)))
      output = get_count(output)
      cv2_imshow(np.reshape(output*255, (sh[1], sh[2])))
      cv2_imshow(np.reshape(y_test*255, (sh[1], sh[2])))

  else:       # Calculate mean and SD for multiple test samples
    cal_avg_metric([prec_list, sens_list, jac_list, dice_list, acc_list])
