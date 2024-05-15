# Function to return the list of input_arr and output dataset for training the model
# 'path' - path to the folder containing the data

def get_dataset(path):

  p = path
  inp_path = p+'images/'
  out_path = p+'labels_1D/'
  l = os.listdir(inp_path)

  inp = []
  out = []
  sparse = []

  for i in l:

    ip = inp_path + i
    op = out_path + i[:-4] + '.png'

    input_arr = cv.imread(ip)
    output = cv.imread(op)

    input_arr = np.array(input_arr, dtype=np.uint8)

    output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    output = np.array(output, dtype=np.uint8)

    input_arr = cv.resize(input_arr, (256, 256), interpolation=cv.INTER_AREA)
    output = cv.resize(output, (256, 256), interpolation=cv.INTER_NEAREST)

    out.append(output)
    inp.append(input_arr)

  inp = np.array(inp, dtype=np.uint8)
  out = np.array(out, dtype=np.uint8)

  return inp, out
