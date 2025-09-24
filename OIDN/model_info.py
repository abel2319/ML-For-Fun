from model import *
from torchinfo import summary

model = UNetLarge(9, 3)
model.eval()
summary(model, input_size=(4, 9, 256, 256))