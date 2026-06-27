from ml.rebuild_cnn import rebuild_cnn

model = rebuild_cnn()

print("Inputs :", model.inputs)
print("Outputs:", model.outputs)

print("\nVerification Successful")