import torch
from movinet import MoViNetA2
import time
torch.manual_seed(1234)
model = MoViNetA2(buffer_size=2,subclip_length=50)

# Generate a random tensor with the required shape
a = torch.rand(2, 3, 65, 224, 224)

def test():
    start_time = time.time()
    output = model(a)
    end_time = time.time()
    print(f"Test execution time: {end_time - start_time} seconds")
    return output

start_time = time.time()
x = model(a)
end_time = time.time()

print(f"Evaluation execution time: {end_time - start_time} seconds")
print(x)
