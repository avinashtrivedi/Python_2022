import numpy as np
import core
import argparse


parser = argparse.ArgumentParser(description='Data generation code')
parser.add_argument('--train', default=1, type=int, help='train or test')
args = parser.parse_args()

if args.train:
    num_unsafe = 100000
    num_safe = 100000
    num_medium = 100000
else:
    num_unsafe = 1000
    num_safe = 1000
    num_medium = 1000

# Sample unsafe examples
unsafe = np.zeros((num_unsafe, 4, 1), dtype=np.float32)
#i = 0
#while True:
#    random_samples = np.random.uniform(low=-1.0, high=1.0, size=(10000, 4, 1))
#    random_samples = random_samples * core.x_max
#    _, _, meta = core.get_safe_mask(random_samples, return_meta=True)
#    inside_bar = random_samples[meta['inside_bar']]
#    unsafe[i:min(num_unsafe//2, i+len(inside_bar))] = inside_bar[:num_unsafe//2-i]
#    i = min(num_unsafe//2, i+len(inside_bar))
#    if i == num_unsafe//2:
#        break

#while True:
#    random_samples = np.random.uniform(low=-1.0, high=1.0, size=(10000, 4, 1))
#    random_samples = random_samples * core.x_max
#    _, _, meta = core.get_safe_mask(random_samples, return_meta=True)
#    tilt_too_much = random_samples[meta['tilt_too_much']]
#    unsafe[i:min(num_unsafe, i+len(tilt_too_much))] = tilt_too_much[:num_unsafe-i]
#    i = min(num_unsafe, i+len(tilt_too_much))
#    if i == num_unsafe:
#        break
i = 0
while True:
    random_samples = np.random.uniform(low=-1.0, high=1.0, size=(10000, 4, 1))
    random_samples = random_samples * core.x_max
    safe_mask, unsafe_mask = core.get_safe_mask(random_samples)
    unsafe_samples = random_samples[unsafe_mask]
    unsafe[i:min(num_unsafe, i+len(unsafe_samples))] = unsafe_samples[:num_unsafe-i]
    i = min(num_unsafe, i+len(unsafe_samples))
    if i == num_unsafe:
        break

# Sample safe examples
safe = np.zeros((num_safe, 4, 1), dtype=np.float32)
i = 0
while True:
    random_samples = np.random.uniform(low=-1.0, high=1.0, size=(10000, 4, 1))
    random_samples = random_samples * core.x_max
    safe_mask, unsafe_mask = core.get_safe_mask(random_samples)
    safe_samples = random_samples[safe_mask]
    safe[i:min(num_safe, i+len(safe_samples))] = safe_samples[:num_safe-i]
    i = min(num_safe, i+len(safe_samples))
    if i == num_safe:
        break

# Sample medium examples
medium = np.zeros((num_safe, 4, 1), dtype=np.float32)
i = 0
while True:
    random_samples = np.random.uniform(low=-1.0, high=1.0, size=(10000, 4, 1))
    random_samples = random_samples * core.x_max
    safe_mask, unsafe_mask = core.get_safe_mask(random_samples)
    medium_mask = np.logical_not(np.logical_or(safe_mask, unsafe_mask))
    medium_samples = random_samples[medium_mask]
    medium[i:min(num_medium, i+len(medium_samples))] = medium_samples[:num_medium-i]
    i = min(num_medium, i+len(medium_samples))
    if i == num_medium:
        break

x = np.concatenate([unsafe, medium, safe], axis=0).astype(np.float16)
y = np.concatenate([-np.ones(num_unsafe), np.zeros(num_medium), np.ones(num_safe)], axis=0)

if args.train:
    f = open('data.npz', 'wb')
else:
    f = open('data_test.npz', 'wb')
np.savez(f, x=x, y=y)