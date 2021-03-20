from lr_utils import load_dataset
from functions import model, check_your_photo

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = 209
m_test = 50
num_px = 64

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

new_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_px, 2000, 0.005, True)

my_image = "obrazek.jpg"

check_your_photo(my_image, num_px, classes, new_model)