from ecg_noise_detector import noiseDetector

a = noiseDetector.get_example_ecg('noisy')
print(a)
# noiseDetector.plot_ecg(a)

print(noiseDetector.is_noisy(a))