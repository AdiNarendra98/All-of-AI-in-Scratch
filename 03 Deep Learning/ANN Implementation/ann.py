{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_cell_guid": "6fe1e8d5-b36d-4e39-9a9b-34618b5e275e",
    "_uuid": "79f18357b846d2cd91e0f7b2389e1dba8097cbdb",
    "execution": {
     "iopub.execute_input": "2022-06-13T05:59:56.904752Z",
     "iopub.status.busy": "2022-06-13T05:59:56.904267Z",
     "iopub.status.idle": "2022-06-13T05:59:56.909824Z",
     "shell.execute_reply": "2022-06-13T05:59:56.909087Z",
     "shell.execute_reply.started": "2022-06-13T05:59:56.904686Z"
    }
   },
   "outputs": [],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load in \n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "import matplotlib.pyplot as plt\n",
    "# Input data files are available in the \"../input/\" directory.\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory\n",
    "# import warnings\n",
    "import warnings\n",
    "# filter warnings\n",
    "warnings.filterwarnings('ignore')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "62bb7abd-122c-4f87-97ad-eb203e9dea1f",
    "_uuid": "3e3e94ca5c9349ac36416482d30af378966c7a8a"
   },
   "source": [
    "<a id=\"Overview the Data Set\"></a> <br>\n",
    "# Overview the Data Set\n",
    "* We will use \"sign language digits data set\" for this tutorial.\n",
    "* In this data there are 2062 sign language digits images.\n",
    "* As you know digits are from 0 to 9. Therefore there are 10 unique sign.\n",
    "* At the beginning of tutorial we will use only sign 0 and 1 for simplicity. \n",
    "* In data, sign zero is between indexes 204 and 408. Number of zero sign is 205.\n",
    "* Also sign one is between indexes 822 and 1027. Number of one sign is 206. Therefore, we will use 205 samples from each classes(labels).\n",
    "* Note: Actually 205 sample is very very very little for deep learning. But this is tutorial so it does not matter so much. \n",
    "* Lets prepare our X and Y arrays. X is image array (zero and one signs) and Y is label array (0 and 1)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "_cell_guid": "4768ce70-3e7f-4ac9-8764-642e88006b77",
    "_uuid": "d6b38399b27c2d723750c0b4f8787a7d6d0025ea",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:00:00.371031Z",
     "iopub.status.busy": "2022-06-13T06:00:00.370501Z",
     "iopub.status.idle": "2022-06-13T06:00:01.413042Z",
     "shell.execute_reply": "2022-06-13T06:00:01.411740Z",
     "shell.execute_reply.started": "2022-06-13T06:00:00.370980Z"
    }
   },
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: '../input/Sign-language-digits-dataset/X.npy'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-2-02560e32ac8f>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# load data set\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mx_l\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'../input/Sign-language-digits-dataset/X.npy'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0mY_l\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'../input/Sign-language-digits-dataset/Y.npy'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mimg_size\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m64\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msubplot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/anaconda3/lib/python3.8/site-packages/numpy/lib/npyio.py\u001b[0m in \u001b[0;36mload\u001b[0;34m(file, mmap_mode, allow_pickle, fix_imports, encoding)\u001b[0m\n\u001b[1;32m    415\u001b[0m             \u001b[0mown_fid\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    416\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 417\u001b[0;31m             \u001b[0mfid\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mstack\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0menter_context\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mos_fspath\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfile\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"rb\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    418\u001b[0m             \u001b[0mown_fid\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    419\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '../input/Sign-language-digits-dataset/X.npy'"
     ]
    }
   ],
   "source": [
    "# load data set\n",
    "x_l = np.load('../input/Sign-language-digits-dataset/X.npy')\n",
    "Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')\n",
    "img_size = 64\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.imshow(x_l[260].reshape(img_size, img_size))\n",
    "plt.axis('off')\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.imshow(x_l[900].reshape(img_size, img_size))\n",
    "plt.axis('off')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "1c9cf8f2-f318-4f69-b92e-9c1bfe78c2c7",
    "_uuid": "c12d1854c6f5fb43c8f081083a441585ed48dc93"
   },
   "source": [
    "* In order to create image array, I concatenate zero sign and one sign arrays\n",
    "* Then I create label array 0 for zero sign images and 1 for one sign images."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "_cell_guid": "88a9b18b-12c7-46d6-8370-e2f02b55dd7a",
    "_uuid": "ad417f6189b0d9476388d92b6be09c887f7f1a40",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:00:09.647949Z",
     "iopub.status.busy": "2022-06-13T06:00:09.647638Z",
     "iopub.status.idle": "2022-06-13T06:00:09.659776Z",
     "shell.execute_reply": "2022-06-13T06:00:09.658735Z",
     "shell.execute_reply.started": "2022-06-13T06:00:09.647903Z"
    }
   },
   "outputs": [],
   "source": [
    "# Join a sequence of arrays along an row axis.\n",
    "X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign \n",
    "z = np.zeros(205)\n",
    "o = np.ones(205)\n",
    "Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)\n",
    "print(\"X shape: \" , X.shape)\n",
    "print(\"Y shape: \" , Y.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "d6024477-cf34-40f9-b079-8388b5146e1b",
    "_uuid": "a999c02e74af603d65a4442842d739aa351f593c"
   },
   "source": [
    "* The shape of the X is (410, 64, 64)\n",
    "    * 410 means that we have 410 images (zero and one signs)\n",
    "    * 64 means that our image size is 64x64 (64x64 pixels)\n",
    "* The shape of the Y is (410,1)\n",
    "    *  410 means that we have 410 labels (0 and 1) \n",
    "* Lets split X and Y into train and test sets.\n",
    "    * test_size = percentage of test size. test = 15% and train = 75%\n",
    "    * random_state = use same seed while randomizing. It means that if we call train_test_split repeatedly, it always creates same train and test distribution because we have same random_state."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "_cell_guid": "00db1f03-75d3-48dc-a6fb-824a58790d42",
    "_uuid": "82869efdb6890ede1899c1bc8c90fa8b1caceec3",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:00:16.840916Z",
     "iopub.status.busy": "2022-06-13T06:00:16.840582Z",
     "iopub.status.idle": "2022-06-13T06:00:17.705439Z",
     "shell.execute_reply": "2022-06-13T06:00:17.703883Z",
     "shell.execute_reply.started": "2022-06-13T06:00:16.840845Z"
    }
   },
   "outputs": [],
   "source": [
    "# Then lets create x_train, y_train, x_test, y_test arrays\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)\n",
    "number_of_train = X_train.shape[0]\n",
    "number_of_test = X_test.shape[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "4bb55e07-cabb-46ce-b21a-fe4a7b34995f",
    "_uuid": "e294dc952e7ba34569162a8965680ead8edd2834"
   },
   "source": [
    "* Now we have 3 dimensional input array (X) so we need to make it flatten (2D) in order to use as input for our first deep learning model.\n",
    "* Our label array (Y) is already flatten(2D) so we leave it like that.\n",
    "* Lets flatten X array(images array).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "_cell_guid": "f5937123-1e16-4036-844b-f498ed42e504",
    "_uuid": "f8b7203144bc85873d960f765e2bb3e711b57f30",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:00:27.446979Z",
     "iopub.status.busy": "2022-06-13T06:00:27.446674Z",
     "iopub.status.idle": "2022-06-13T06:00:27.452720Z",
     "shell.execute_reply": "2022-06-13T06:00:27.451844Z",
     "shell.execute_reply.started": "2022-06-13T06:00:27.446935Z"
    }
   },
   "outputs": [],
   "source": [
    "X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])\n",
    "X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])\n",
    "print(\"X train flatten\",X_train_flatten.shape)\n",
    "print(\"X test flatten\",X_test_flatten.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "dab40a5b-e319-4807-a608-12065d860847",
    "_uuid": "cb26fc242ffcae8bf8f164d32dc9c5436e8312fc"
   },
   "source": [
    "* As you can see, we have 348 images and each image has 4096 pixels in image train array.\n",
    "* Also, we have 62 images and each image has 4096 pixels in image test array.\n",
    "* Then lets take transpose. You can say that WHYY, actually there is no technical answer. I just write the code(code that you will see oncoming parts) according to it :)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "_cell_guid": "ad9bee66-78f1-44ec-a114-465356b9cc7d",
    "_uuid": "88eef1b839ee51234a53da3ccb1e36a2e5e9a0e6",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:00:30.416708Z",
     "iopub.status.busy": "2022-06-13T06:00:30.416045Z",
     "iopub.status.idle": "2022-06-13T06:00:30.424878Z",
     "shell.execute_reply": "2022-06-13T06:00:30.424164Z",
     "shell.execute_reply.started": "2022-06-13T06:00:30.416639Z"
    }
   },
   "outputs": [],
   "source": [
    "x_train = X_train_flatten.T\n",
    "x_test = X_test_flatten.T\n",
    "y_train = Y_train.T\n",
    "y_test = Y_test.T\n",
    "print(\"x train: \",x_train.shape)\n",
    "print(\"x test: \",x_test.shape)\n",
    "print(\"y train: \",y_train.shape)\n",
    "print(\"y test: \",y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "8b5f8812-f21c-4936-91b9-d33048dec40b",
    "_uuid": "5f037b5d3f44a9bf139dddb7946c0b33cf7d0298"
   },
   "source": [
    "<a id=\"3\"></a> <br>\n",
    "# Logistic Regression\n",
    "* When we talk about binary classification( 0 and 1 outputs) what comes to mind first is logistic regression.\n",
    "* However, in deep learning tutorial what to do with logistic regression there??\n",
    "* The answer is that  logistic regression is actually a very simple neural network. \n",
    "* By the way neural network and deep learning are same thing. When we will come artificial neural network, I will explain detailed the terms like \"deep\".\n",
    "* In order to understand logistic regression (simple deep learning) lets first learn computation graph."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "b9ec7e1d-186b-4911-be80-bc856f43b689",
    "_uuid": "40c4a1af2372960d36b6649f5baf4915e9e16738"
   },
   "source": [
    "<a id=\"5\"></a> <br>\n",
    "## Initializing parameters\n",
    "* As you know input is our images that has 4096 pixels(each image in x_train).\n",
    "* Each pixels have own weights.\n",
    "* The first step is multiplying each pixels with their own weights.\n",
    "* The question is that what is the initial value of weights?\n",
    "    * There are some techniques that I will explain at artificial neural network but for this time initial weights are 0.01.\n",
    "    * Okey, weights are 0.01 but what is the weight array shape? As you understand from computation graph of logistic regression, it is (4096,1)\n",
    "    * Also initial bias is 0.\n",
    "* Lets write some code. In order to use at coming topics like artificial neural network (ANN), I make definition(method)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "_cell_guid": "74d461fc-4aa9-4b76-bd26-43551cef138b",
    "_uuid": "f3be95b8ca86fea08336badabf563b5f8f59a142",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:00:53.789955Z",
     "iopub.status.busy": "2022-06-13T06:00:53.789373Z",
     "iopub.status.idle": "2022-06-13T06:00:53.794764Z",
     "shell.execute_reply": "2022-06-13T06:00:53.794128Z",
     "shell.execute_reply.started": "2022-06-13T06:00:53.789886Z"
    }
   },
   "outputs": [],
   "source": [
    "# short description and example of definition (def)\n",
    "def dummy(parameter):\n",
    "    dummy_parameter = parameter + 5\n",
    "    return dummy_parameter\n",
    "result = dummy(3)     # result = 8\n",
    "\n",
    "# lets initialize parameters\n",
    "# So what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)\n",
    "def initialize_weights_and_bias(dimension):\n",
    "    w = np.full((dimension,1),0.01)\n",
    "    b = 0.0\n",
    "    return w, b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_uuid": "d71d5b420ee146833d479781ea889eeebcb74e47"
   },
   "outputs": [],
   "source": [
    "#w,b = initialize_weights_and_bias(4096)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "1fbf27d5-1f4b-4b94-a757-99e2becf54a0",
    "_uuid": "dde6fab55966b2de3dc808e03613df9bea7bb0d2"
   },
   "source": [
    "<a id=\"6\"></a> <br>\n",
    "## Forward Propagation\n",
    "* The all steps from pixels to cost is called forward propagation\n",
    "    * z = (w.T)x + b => in this equation we know x that is pixel array, we know w (weights) and b (bias) so the rest is calculation. (T is transpose)\n",
    "    * Then we put z into sigmoid function that returns y_head(probability). When your mind is confused go and look at computation graph. Also equation of sigmoid function is in computation graph.\n",
    "    * Then we calculate loss(error) function. \n",
    "    * Cost function is summation of all loss(error).\n",
    "    * Lets start with z and the write sigmoid definition(method) that takes z as input parameter and returns y_head(probability)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "_cell_guid": "697be401-792b-46c0-8fe6-79cd65110419",
    "_uuid": "e024479bce9ce2022f65ffd586a6c29b124e7ab5",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:01:00.309210Z",
     "iopub.status.busy": "2022-06-13T06:01:00.308828Z",
     "iopub.status.idle": "2022-06-13T06:01:00.313434Z",
     "shell.execute_reply": "2022-06-13T06:01:00.312672Z",
     "shell.execute_reply.started": "2022-06-13T06:01:00.309147Z"
    }
   },
   "outputs": [],
   "source": [
    "# calculation of z\n",
    "#z = np.dot(w.T,x_train)+b\n",
    "def sigmoid(z):\n",
    "    y_head = 1/(1+np.exp(-z))\n",
    "    return y_head"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "_uuid": "da7e0244449200eaf382507806f01901f1d281c8",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:01:02.206672Z",
     "iopub.status.busy": "2022-06-13T06:01:02.206297Z",
     "iopub.status.idle": "2022-06-13T06:01:02.213662Z",
     "shell.execute_reply": "2022-06-13T06:01:02.212753Z",
     "shell.execute_reply.started": "2022-06-13T06:01:02.206618Z"
    }
   },
   "outputs": [],
   "source": [
    "y_head = sigmoid(0)\n",
    "y_head"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "571dc02a-b25d-4726-ad2b-9661ae6783e2",
    "_uuid": "d67ca31d01cc8f03fdd862789454e4acf1b2033b"
   },
   "source": [
    "* As we write sigmoid method and calculate y_head. Lets learn what is loss(error) function\n",
    "* Lets make example, I put one image as input then multiply it with their weights and add bias term so I find z. Then put z into sigmoid method so I find y_head. Up to this point we know what we did. Then e.g y_head became 0.9 that is bigger than 0.5 so our prediction is image is sign one image. Okey every thing looks like fine. But, is our prediction is correct and how do we check whether it is correct or not? The answer is with loss(error) function:\n",
    "    * Mathematical expression of log loss(error) function is that: \n",
    "    <a href=\"https://imgbb.com/\"><img src=\"https://image.ibb.co/eC0JCK/duzeltme.jpg\" alt=\"duzeltme\" border=\"0\"></a>\n",
    "    * It says that if you make wrong prediction, loss(error) becomes big. **DENKLEM DUZELTME**\n",
    "        * Example: our real image is sign one and its label is 1 (y = 1), then we make prediction y_head = 1. When we put y and y_head into loss(error) equation the result is 0. We make correct prediction therefore our loss is 0. However, if we make wrong prediction like y_head = 0, loss(error) is infinity.\n",
    "* After that, the cost function is summation of loss function. Each image creates loss function. Cost function is summation of loss functions that is created by each input image.\n",
    "* Lets implement forward propagation.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "_cell_guid": "adbc7d22-8bba-48c1-b754-75c9d4f0543c",
    "_uuid": "447f1ee51819fdfeba0e1ce0b03449cb549510ed",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:01:05.517660Z",
     "iopub.status.busy": "2022-06-13T06:01:05.516771Z",
     "iopub.status.idle": "2022-06-13T06:01:05.523991Z",
     "shell.execute_reply": "2022-06-13T06:01:05.523387Z",
     "shell.execute_reply.started": "2022-06-13T06:01:05.517561Z"
    }
   },
   "outputs": [],
   "source": [
    "# Forward propagation steps:\n",
    "# find z = w.T*x+b\n",
    "# y_head = sigmoid(z)\n",
    "# loss(error) = loss(y,y_head)\n",
    "# cost = sum(loss)\n",
    "def forward_propagation(w,b,x_train,y_train):\n",
    "    z = np.dot(w.T,x_train) + b\n",
    "    y_head = sigmoid(z) # probabilistic 0-1\n",
    "    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)\n",
    "    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling\n",
    "    return cost "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "_cell_guid": "2b74bb7c-bcbe-4865-ac61-25ced274e7de",
    "_uuid": "088f192f99d2a31f041116c893ec713ef3b4fe43",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:01:30.637936Z",
     "iopub.status.busy": "2022-06-13T06:01:30.637605Z",
     "iopub.status.idle": "2022-06-13T06:01:30.645118Z",
     "shell.execute_reply": "2022-06-13T06:01:30.644212Z",
     "shell.execute_reply.started": "2022-06-13T06:01:30.637874Z"
    }
   },
   "outputs": [],
   "source": [
    "# In backward propagation we will use y_head that found in forward progation\n",
    "# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation\n",
    "def forward_backward_propagation(w,b,x_train,y_train):\n",
    "    # forward propagation\n",
    "    z = np.dot(w.T,x_train) + b\n",
    "    y_head = sigmoid(z)\n",
    "    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)\n",
    "    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling\n",
    "    # backward propagation\n",
    "    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling\n",
    "    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling\n",
    "    gradients = {\"derivative_weight\": derivative_weight,\"derivative_bias\": derivative_bias}\n",
    "    return cost,gradients"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "d82dbae8-d11e-4ea8-bbfe-545fee3166b3",
    "_uuid": "9e4d028259897e1565341736d58e46f70ea6b312"
   },
   "source": [
    "* Up to this point we learn \n",
    "    * Initializing parameters (implemented)\n",
    "    * Finding cost with forward propagation and cost function (implemented)\n",
    "    * Updating(learning) parameters (weight and bias). Now lets implement it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "_cell_guid": "0940e18d-636d-4503-a2c6-1994024726fe",
    "_uuid": "31299bda686ae2ab157ea18df70e4741e5225345",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:01:32.443880Z",
     "iopub.status.busy": "2022-06-13T06:01:32.443384Z",
     "iopub.status.idle": "2022-06-13T06:01:32.452502Z",
     "shell.execute_reply": "2022-06-13T06:01:32.451835Z",
     "shell.execute_reply.started": "2022-06-13T06:01:32.443828Z"
    }
   },
   "outputs": [],
   "source": [
    "# Updating(learning) parameters\n",
    "def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):\n",
    "    cost_list = []\n",
    "    cost_list2 = []\n",
    "    index = []\n",
    "    # updating(learning) parameters is number_of_iterarion times\n",
    "    for i in range(number_of_iterarion):\n",
    "        # make forward and backward propagation and find cost and gradients\n",
    "        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)\n",
    "        cost_list.append(cost)\n",
    "        # lets update\n",
    "        w = w - learning_rate * gradients[\"derivative_weight\"]\n",
    "        b = b - learning_rate * gradients[\"derivative_bias\"]\n",
    "        if i % 10 == 0:\n",
    "            cost_list2.append(cost)\n",
    "            index.append(i)\n",
    "            print (\"Cost after iteration %i: %f\" %(i, cost))\n",
    "    # we update(learn) parameters weights and bias\n",
    "    parameters = {\"weight\": w,\"bias\": b}\n",
    "    plt.plot(index,cost_list2)\n",
    "    plt.xticks(index,rotation='vertical')\n",
    "    plt.xlabel(\"Number of Iterarion\")\n",
    "    plt.ylabel(\"Cost\")\n",
    "    plt.show()\n",
    "    return parameters, gradients, cost_list\n",
    "#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "202a05c6-2187-40eb-9733-e4da792f04b6",
    "_uuid": "1892ccefd7debe1e9f59b500873b7afded3a4a01"
   },
   "source": [
    "* Woow, I get tired :) Up to this point we learn our parameters. It means we fit the data. \n",
    "* In order to predict we have parameters. Therefore, lets predict.\n",
    "* In prediction step we have x_test as a input and while using it, we make forward prediction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "_cell_guid": "1fd35d0f-e989-49fa-9bcb-1769d92a5ab4",
    "_uuid": "c9b80f081f49a722818cfebfc01e8b8dc69db9c1",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:01:33.925394Z",
     "iopub.status.busy": "2022-06-13T06:01:33.924925Z",
     "iopub.status.idle": "2022-06-13T06:01:33.931942Z",
     "shell.execute_reply": "2022-06-13T06:01:33.931322Z",
     "shell.execute_reply.started": "2022-06-13T06:01:33.925347Z"
    }
   },
   "outputs": [],
   "source": [
    " # prediction\n",
    "def predict(w,b,x_test):\n",
    "    # x_test is a input for forward propagation\n",
    "    z = sigmoid(np.dot(w.T,x_test)+b)\n",
    "    Y_prediction = np.zeros((1,x_test.shape[1]))\n",
    "    # if z is bigger than 0.5, our prediction is sign one (y_head=1),\n",
    "    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),\n",
    "    for i in range(z.shape[1]):\n",
    "        if z[0,i]<= 0.5:\n",
    "            Y_prediction[0,i] = 0\n",
    "        else:\n",
    "            Y_prediction[0,i] = 1\n",
    "\n",
    "    return Y_prediction\n",
    "# predict(parameters[\"weight\"],parameters[\"bias\"],x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "ce815319-161b-408b-84fe-4bb1fab92d13",
    "_uuid": "40dbb73794b6b742b01e33038284aa9f11cc698c"
   },
   "source": [
    "* We make prediction.\n",
    "* Now lets put them all together."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "_cell_guid": "029e2cd3-125b-4ca4-8b94-25d8164ba1a7",
    "_uuid": "81fb6989ff3860d72462f8212b1d00325272a471",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:01:37.461424Z",
     "iopub.status.busy": "2022-06-13T06:01:37.460924Z",
     "iopub.status.idle": "2022-06-13T06:01:38.389525Z",
     "shell.execute_reply": "2022-06-13T06:01:38.388612Z",
     "shell.execute_reply.started": "2022-06-13T06:01:37.461369Z"
    }
   },
   "outputs": [],
   "source": [
    "def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):\n",
    "    # initialize\n",
    "    dimension =  x_train.shape[0]  # that is 4096\n",
    "    w,b = initialize_weights_and_bias(dimension)\n",
    "    # do not change learning rate\n",
    "    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)\n",
    "    \n",
    "    y_prediction_test = predict(parameters[\"weight\"],parameters[\"bias\"],x_test)\n",
    "    y_prediction_train = predict(parameters[\"weight\"],parameters[\"bias\"],x_train)\n",
    "\n",
    "    # Print train/test Errors\n",
    "    print(\"train accuracy: {} %\".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))\n",
    "    print(\"test accuracy: {} %\".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))\n",
    "    \n",
    "logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "6603c5a2-0a1b-4e4a-addd-a89be453e6fd",
    "_uuid": "1da3b972bab3207d2ba77f3d21516296f3e9be02"
   },
   "source": [
    "* We learn logic behind simple neural network(logistic regression) and how to implement it.\n",
    "* Now that we have learned logic, we can use sklearn library which is easier than implementing all steps with hand for logistic regression.\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "119db4ab-f04c-41c0-aef9-728565786e94",
    "_uuid": "ba8b9c960d1735b146fbbad378c90902e7218d59"
   },
   "source": [
    "<a id=\"8\"></a> <br>\n",
    "## Logistic Regression with Sklearn\n",
    "* In sklearn library, there is a logistic regression method that ease implementing logistic regression.\n",
    "* I am not going to explain each parameter of logistic regression in sklear but if you want you can read from there http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html\n",
    "* The accuracies are different from what we find. Because logistic regression method use a lot of different feature that we do not use like different optimization parameters or regularization.\n",
    "* Lets make conclusion for logistic regression and continue with artificial neural network."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "_cell_guid": "5bc37200-cd53-4da3-a2ef-c630cf1a2d10",
    "_uuid": "a13b1565f8a5ae234eaa9ed6da288103a3579938",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:01:44.844852Z",
     "iopub.status.busy": "2022-06-13T06:01:44.844506Z",
     "iopub.status.idle": "2022-06-13T06:01:45.514103Z",
     "shell.execute_reply": "2022-06-13T06:01:45.511397Z",
     "shell.execute_reply.started": "2022-06-13T06:01:44.844806Z"
    }
   },
   "outputs": [],
   "source": [
    "from sklearn import linear_model\n",
    "logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)\n",
    "print(\"test accuracy: {} \".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))\n",
    "print(\"train accuracy: {} \".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "a800aa00-7adf-4d10-83c1-3a5221219329",
    "_uuid": "dd0e970ce5194a34ca55b27b6580ef85714e1b92"
   },
   "source": [
    "<a id=\"10\"></a> <br>\n",
    "# Artificial Neural Network (ANN)\n",
    "* It is also called deep neural network or deep learning.\n",
    "* **What is neural network:** It is basically taking logistic regression and repeating it at least 2 times.\n",
    "* In logistic regression, there are input and output layers. However, in neural network, there is at least one hidden layer between input and output layer.\n",
    "* **What is deep, in order to say \"deep\" how many layer do I need to have:** When I ask this question to my teacher, he said that \"\"Deep\" is a relative term; it of course refers to the \"depth\" of a network, meaning how many hidden layers it has. \"How deep is your swimming pool?\" could be 12 feet or it might be two feet; nevertheless, it still has a depth--it has the quality of \"deepness\". 32 years ago, I used two or three hidden layers. That was the limit for the specialized hardware of the day. Just a few years ago, 20 layers was considered pretty deep. In October, Andrew Ng mentioned 152 layers was (one of?) the biggest commercial networks he knew of. Last week, I talked to someone at a big, famous company who said he was using \"thousands\". So I prefer to just stick with \"How deep?\"\"\n",
    "* **Why it is called hidden:** Because hidden layer does not see inputs(training set)\n",
    "* For example you have input, one hidden and output layers. When someone ask you \"hey my friend how many layers do your neural network have?\" The answer is \"I have 2 layer neural network\". Because while computing layer number input layer is ignored. \n",
    "* Lets see 2 layer neural network: \n",
    "<a href=\"http://ibb.co/eF315x\"><img src=\"http://preview.ibb.co/dajVyH/9.jpg\" alt=\"9\" border=\"0\"></a>\n",
    "* Step by step we will learn this image.\n",
    "    * As you can see there is one hidden layer between input and output layers. And this hidden layer has 3 nodes. If yoube curious why I choose number of node 3, the answer is there is no reason, I only choose :). Number of node is hyperparameter like learning rate. Therefore we will see hyperparameters at the end of artificial neural network.\n",
    "    * Input and output layers do not change. They are same like logistic regression.\n",
    "    * In image, there is a tanh function that is unknown for you. It is a activation function like sigmoid function. Tanh activation function is better than sigmoid for hidden units bacause mean of its output is closer to zero so it centers the data better for the next layer. Also tanh activation function increase non linearity that cause our model learning better.\n",
    "    * As you can see with purple color there are two parts. Both parts are like logistic regression. The only difference is activation function, inputs and outputs.\n",
    "        * In logistic regression: input => output\n",
    "        * In 2 layer neural network: input => hidden layer => output. You can think that hidden layer is output of part 1 and input of part 2.\n",
    "* Thats all. We will follow the same path like logistic regression for 2 layer neural network.\n",
    "   \n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "35b73665-27cb-48a7-abb1-0e0e6bc10ec0",
    "_uuid": "689a4a049579fff0c718bc30639f18e2dcab8ccc"
   },
   "source": [
    "<a id=\"11\"></a> <br>\n",
    "## 2-Layer Neural Network\n",
    "* Size of layers and initializing parameters weights and bias\n",
    "* Forward propagation\n",
    "* Loss function and Cost function\n",
    "* Backward propagation\n",
    "* Update Parameters\n",
    "* Prediction with learnt parameters weight and bias\n",
    "* Create Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "bbfe8df0-bf38-4842-aec1-c04417a99420",
    "_uuid": "0a5e286cf360d579eb6e5d5f220dd1a17c458039"
   },
   "source": [
    "<a id=\"12\"></a> <br>\n",
    "## Size of layers and initializing parameters weights and bias\n",
    "* For x_train that has 348 sample $x^{(348)}$:\n",
    "$$z^{[1] (348)} =  W^{[1]} x^{(348)} + b^{[1] (348)}$$ \n",
    "$$a^{[1] (348)} = \\tanh(z^{[1] (348)})$$\n",
    "$$z^{[2] (348)} = W^{[2]} a^{[1] (348)} + b^{[2] (348)}$$\n",
    "$$\\hat{y}^{(348)} = a^{[2] (348)} = \\sigma(z^{ [2] (348)})$$\n",
    "\n",
    "* At logistic regression, we initialize weights 0.01 and bias 0. At this time, we initialize weights randomly. Because if we initialize parameters zero each neuron in the first hidden layer will perform the same comptation. Therefore, even after multiple iterartion of gradiet descent each neuron in the layer will be computing same things as other neurons. Therefore we initialize randomly. Also initial weights will be small. If they are very large initially, this will cause the inputs of the tanh to be very large, thus causing gradients to be close to zero. The optimization algorithm will be slow.\n",
    "* Bias can be zero initially."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "_cell_guid": "089fd577-95a0-4218-9b53-72bf1c0ab206",
    "_uuid": "922670a74f6999885759399ebea8b10692796a29",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:02:12.938573Z",
     "iopub.status.busy": "2022-06-13T06:02:12.938234Z",
     "iopub.status.idle": "2022-06-13T06:02:12.944874Z",
     "shell.execute_reply": "2022-06-13T06:02:12.944034Z",
     "shell.execute_reply.started": "2022-06-13T06:02:12.938514Z"
    }
   },
   "outputs": [],
   "source": [
    "# intialize parameters and layer sizes\n",
    "def initialize_parameters_and_layer_sizes_NN(x_train, y_train):\n",
    "    parameters = {\"weight1\": np.random.randn(3,x_train.shape[0]) * 0.1,\n",
    "                  \"bias1\": np.zeros((3,1)),\n",
    "                  \"weight2\": np.random.randn(y_train.shape[0],3) * 0.1,\n",
    "                  \"bias2\": np.zeros((y_train.shape[0],1))}\n",
    "    return parameters"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "65832cdf-7ee8-447b-a068-48ac2e46b49f",
    "_uuid": "66147bbafbe25dac498f3963ea8419126f624ce9"
   },
   "source": [
    "<a id=\"13\"></a> <br>\n",
    "## Forward propagation\n",
    "* Forward propagation is almost same with logistic regression.\n",
    "* The only difference is we use tanh function and we make all process twice.\n",
    "* Also numpy has tanh function. So we do not need to implement it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "_cell_guid": "d64d6b90-7f14-453f-a401-4119504496e3",
    "_uuid": "41e1e2f1c7afff027ba0a9f9b2fdcbe312e9a194",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:02:16.163275Z",
     "iopub.status.busy": "2022-06-13T06:02:16.162727Z",
     "iopub.status.idle": "2022-06-13T06:02:16.168494Z",
     "shell.execute_reply": "2022-06-13T06:02:16.167591Z",
     "shell.execute_reply.started": "2022-06-13T06:02:16.163204Z"
    }
   },
   "outputs": [],
   "source": [
    "\n",
    "def forward_propagation_NN(x_train, parameters):\n",
    "\n",
    "    Z1 = np.dot(parameters[\"weight1\"],x_train) +parameters[\"bias1\"]\n",
    "    A1 = np.tanh(Z1)\n",
    "    Z2 = np.dot(parameters[\"weight2\"],A1) + parameters[\"bias2\"]\n",
    "    A2 = sigmoid(Z2)\n",
    "\n",
    "    cache = {\"Z1\": Z1,\n",
    "             \"A1\": A1,\n",
    "             \"Z2\": Z2,\n",
    "             \"A2\": A2}\n",
    "    \n",
    "    return A2, cache\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "30a0abc9-7ee0-4093-afd5-ae9d5b1fcd5e",
    "_uuid": "ee7a42ee207e222eed5c24b1bfbf2d6ce0cdec37"
   },
   "source": [
    "<a id=\"14\"></a> <br>\n",
    "## Loss function and Cost function\n",
    "* Loss and cost functions are same with logistic regression\n",
    "* Cross entropy function\n",
    "<a href=\"https://imgbb.com/\"><img src=\"https://image.ibb.co/nyR9LU/as.jpg\" alt=\"as\" border=\"0\"></a><br />"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "_cell_guid": "24143d72-bc62-4f2d-b0cf-4f67a4016299",
    "_uuid": "b55887b28cffc8083a76af45d25957d4f3e9f6fa",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:02:19.814006Z",
     "iopub.status.busy": "2022-06-13T06:02:19.813453Z",
     "iopub.status.idle": "2022-06-13T06:02:19.818191Z",
     "shell.execute_reply": "2022-06-13T06:02:19.817329Z",
     "shell.execute_reply.started": "2022-06-13T06:02:19.813941Z"
    }
   },
   "outputs": [],
   "source": [
    "# Compute cost\n",
    "def compute_cost_NN(A2, Y, parameters):\n",
    "    logprobs = np.multiply(np.log(A2),Y)\n",
    "    cost = -np.sum(logprobs)/Y.shape[1]\n",
    "    return cost\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "39839772-976a-4e53-a2e0-07aa76c9bd98",
    "_uuid": "43767f9271e2b2b6e0c3560e414f3c0c596ffe2f"
   },
   "source": [
    "<a id=\"15\"></a> <br>\n",
    "## Backward propagation\n",
    "* As you know backward propagation means derivative.\n",
    "* If you want to learn (as I said I cannot explain without talking bc it is little confusing), please watch video in youtube.\n",
    "* However the logic is same, lets write code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "_cell_guid": "2fcfd3c4-f935-4272-a284-c2dbc2c35afb",
    "_uuid": "6bf7bce2e4413ecdc16ea778008eee4072738aab",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:02:33.321833Z",
     "iopub.status.busy": "2022-06-13T06:02:33.321205Z",
     "iopub.status.idle": "2022-06-13T06:02:33.328631Z",
     "shell.execute_reply": "2022-06-13T06:02:33.327635Z",
     "shell.execute_reply.started": "2022-06-13T06:02:33.321777Z"
    }
   },
   "outputs": [],
   "source": [
    "# Backward Propagation\n",
    "def backward_propagation_NN(parameters, cache, X, Y):\n",
    "\n",
    "    dZ2 = cache[\"A2\"]-Y\n",
    "    dW2 = np.dot(dZ2,cache[\"A1\"].T)/X.shape[1]\n",
    "    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]\n",
    "    dZ1 = np.dot(parameters[\"weight2\"].T,dZ2)*(1 - np.power(cache[\"A1\"], 2))\n",
    "    dW1 = np.dot(dZ1,X.T)/X.shape[1]\n",
    "    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]\n",
    "    grads = {\"dweight1\": dW1,\n",
    "             \"dbias1\": db1,\n",
    "             \"dweight2\": dW2,\n",
    "             \"dbias2\": db2}\n",
    "    return grads"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "af195fda-5649-4e6d-830e-72123f2726a8",
    "_uuid": "b1996782dc44fda7993407c9b5efee5d4fef46e4"
   },
   "source": [
    "<a id=\"16\"></a> <br>\n",
    "## Update Parameters \n",
    "* Updating parameters also same with logistic regression.\n",
    "* We actually do alot of work with logistic regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "_cell_guid": "d9ae95d4-1d11-4293-822d-e1d5f0c16d1e",
    "_uuid": "facf2b475cb82e14dcc6be56b57fe2c3ad0b1a8f",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:02:36.696597Z",
     "iopub.status.busy": "2022-06-13T06:02:36.696244Z",
     "iopub.status.idle": "2022-06-13T06:02:36.702625Z",
     "shell.execute_reply": "2022-06-13T06:02:36.701732Z",
     "shell.execute_reply.started": "2022-06-13T06:02:36.696533Z"
    }
   },
   "outputs": [],
   "source": [
    "# update parameters\n",
    "def update_parameters_NN(parameters, grads, learning_rate = 0.01):\n",
    "    parameters = {\"weight1\": parameters[\"weight1\"]-learning_rate*grads[\"dweight1\"],\n",
    "                  \"bias1\": parameters[\"bias1\"]-learning_rate*grads[\"dbias1\"],\n",
    "                  \"weight2\": parameters[\"weight2\"]-learning_rate*grads[\"dweight2\"],\n",
    "                  \"bias2\": parameters[\"bias2\"]-learning_rate*grads[\"dbias2\"]}\n",
    "    \n",
    "    return parameters"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "ac416480-ec9c-45b4-ac9d-1caeded9ba90",
    "_uuid": "9c471502563017fabb991494359091215e4ad583"
   },
   "source": [
    "<a id=\"17\"></a> <br>\n",
    "## Prediction with learnt parameters weight and bias\n",
    "* Lets write predict method that is like logistic regression."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "_cell_guid": "96004eb5-d6ca-41ab-a577-70fb0628a2f4",
    "_uuid": "53c00c4430c6fcc3298dde8de804cab71884caa5",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:02:42.044139Z",
     "iopub.status.busy": "2022-06-13T06:02:42.043489Z",
     "iopub.status.idle": "2022-06-13T06:02:42.049289Z",
     "shell.execute_reply": "2022-06-13T06:02:42.048364Z",
     "shell.execute_reply.started": "2022-06-13T06:02:42.044068Z"
    }
   },
   "outputs": [],
   "source": [
    "# prediction\n",
    "def predict_NN(parameters,x_test):\n",
    "    # x_test is a input for forward propagation\n",
    "    A2, cache = forward_propagation_NN(x_test,parameters)\n",
    "    Y_prediction = np.zeros((1,x_test.shape[1]))\n",
    "    # if z is bigger than 0.5, our prediction is sign one (y_head=1),\n",
    "    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),\n",
    "    for i in range(A2.shape[1]):\n",
    "        if A2[0,i]<= 0.5:\n",
    "            Y_prediction[0,i] = 0\n",
    "        else:\n",
    "            Y_prediction[0,i] = 1\n",
    "\n",
    "    return Y_prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "d0df9e13-300b-4d5e-b8ec-f702ed0e06af",
    "_uuid": "94202fbc047d59fa5c8b81ba02962f1f3cc56d8f"
   },
   "source": [
    "<a id=\"18\"></a> <br>\n",
    "## Create Model\n",
    "* Lets put them all together"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "_cell_guid": "b66f3c28-0f71-4176-8a6b-35b98b0db936",
    "_uuid": "9babf239f800bedc9864c6a75677985d57f1cc78",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:03:06.281553Z",
     "iopub.status.busy": "2022-06-13T06:03:06.280610Z",
     "iopub.status.idle": "2022-06-13T06:03:21.601285Z",
     "shell.execute_reply": "2022-06-13T06:03:21.600325Z",
     "shell.execute_reply.started": "2022-06-13T06:03:06.281490Z"
    }
   },
   "outputs": [],
   "source": [
    "# 2 - Layer neural network\n",
    "def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):\n",
    "    cost_list = []\n",
    "    index_list = []\n",
    "    #initialize parameters and layer sizes\n",
    "    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)\n",
    "\n",
    "    for i in range(0, num_iterations):\n",
    "         # forward propagation\n",
    "        A2, cache = forward_propagation_NN(x_train,parameters)\n",
    "        # compute cost\n",
    "        cost = compute_cost_NN(A2, y_train, parameters)\n",
    "         # backward propagation\n",
    "        grads = backward_propagation_NN(parameters, cache, x_train, y_train)\n",
    "         # update parameters\n",
    "        parameters = update_parameters_NN(parameters, grads)\n",
    "        \n",
    "        if i % 100 == 0:\n",
    "            cost_list.append(cost)\n",
    "            index_list.append(i)\n",
    "            print (\"Cost after iteration %i: %f\" %(i, cost))\n",
    "    plt.plot(index_list,cost_list)\n",
    "    plt.xticks(index_list,rotation='vertical')\n",
    "    plt.xlabel(\"Number of Iterarion\")\n",
    "    plt.ylabel(\"Cost\")\n",
    "    plt.show()\n",
    "    \n",
    "    # predict\n",
    "    y_prediction_test = predict_NN(parameters,x_test)\n",
    "    y_prediction_train = predict_NN(parameters,x_train)\n",
    "\n",
    "    # Print train/test Errors\n",
    "    print(\"train accuracy: {} %\".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))\n",
    "    print(\"test accuracy: {} %\".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))\n",
    "    return parameters\n",
    "\n",
    "parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "04d3b704-2fd4-410b-ad90-a5a0aedd4b7d",
    "_uuid": "62bb69efcf883ba22581dfc8d4f2e0eaa2d99ee3"
   },
   "source": [
    "<font color='purple'>\n",
    "Up to this point we create 2 layer neural network and learn how to implement\n",
    "* Size of layers and initializing parameters weights and bias\n",
    "* Forward propagation\n",
    "* Loss function and Cost function\n",
    "* Backward propagation\n",
    "* Update Parameters\n",
    "* Prediction with learnt parameters weight and bias\n",
    "* Create Model\n",
    "\n",
    "<br> Now lets learn how to implement L layer neural network with keras."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "_cell_guid": "631a05c4-e362-4fa0-9048-21c599f55344",
    "_uuid": "0a978924a68d423de4babe73c15412ad938c1858",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:03:26.747431Z",
     "iopub.status.busy": "2022-06-13T06:03:26.747063Z",
     "iopub.status.idle": "2022-06-13T06:03:26.751589Z",
     "shell.execute_reply": "2022-06-13T06:03:26.750660Z",
     "shell.execute_reply.started": "2022-06-13T06:03:26.747378Z"
    }
   },
   "outputs": [],
   "source": [
    "# reshaping\n",
    "x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "_cell_guid": "e17b5a34-00dc-49a8-b7ac-a2bd3a753f78",
    "_uuid": "5a78a5570bc50180a02190dee46180f19eb165e1"
   },
   "source": [
    "<a id=\"22\"></a> <br>\n",
    "## Implementing with keras library\n",
    "Lets look at some parameters of keras library:\n",
    "* units: output dimensions of node\n",
    "* kernel_initializer: to initialize weights\n",
    "* activation: activation function, we use relu\n",
    "* input_dim: input dimension that is number of pixels in our images (4096 px)\n",
    "* optimizer: we use adam optimizer\n",
    "    * Adam is one of the most effective optimization algorithms for training neural networks.\n",
    "    * Some advantages of Adam is that relatively low memory requirements and usually works well even with little tuning of hyperparameters\n",
    "* loss: Cost function is same. By the way the name of the cost function is cross-entropy cost function that we use previous parts.\n",
    "$$J = - \\frac{1}{m} \\sum\\limits_{i = 0}^{m} \\large\\left(\\small y^{(i)}\\log\\left(a^{[2] (i)}\\right) + (1-y^{(i)})\\log\\left(1- a^{[2] (i)}\\right)  \\large  \\right) \\small \\tag{6}$$\n",
    "* metrics: it is accuracy.\n",
    "* cross_val_score: use cross validation. If you do not know cross validation please chech it from my machine learning tutorial. https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners\n",
    "* epochs: number of iteration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "_cell_guid": "8870c45b-b0fe-4050-a8af-41498a417ed5",
    "_uuid": "9361c3183a40fa0080055b7d5c1002aef68b4d77",
    "execution": {
     "iopub.execute_input": "2022-06-13T06:03:39.363183Z",
     "iopub.status.busy": "2022-06-13T06:03:39.362679Z",
     "iopub.status.idle": "2022-06-13T06:03:48.729650Z",
     "shell.execute_reply": "2022-06-13T06:03:48.728822Z",
     "shell.execute_reply.started": "2022-06-13T06:03:39.363143Z"
    }
   },
   "outputs": [],
   "source": [
    "# Evaluating the ANN\n",
    "from keras.wrappers.scikit_learn import KerasClassifier\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from keras.models import Sequential # initialize neural network library\n",
    "from keras.layers import Dense # build our layers library\n",
    "def build_classifier():\n",
    "    classifier = Sequential() # initialize neural network\n",
    "    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))\n",
    "    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))\n",
    "    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))\n",
    "    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])\n",
    "    return classifier\n",
    "classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)\n",
    "accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)\n",
    "mean = accuracies.mean()\n",
    "variance = accuracies.std()\n",
    "print(\"Accuracy mean: \"+ str(mean))\n",
    "print(\"Accuracy variance: \"+ str(variance))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_uuid": "59ad9da159449a3ed9e99ed0ec931ee14b7aab66"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
