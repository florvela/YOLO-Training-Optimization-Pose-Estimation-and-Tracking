from tvm.driver import tvmc
import tvm

model = tvmc.load('my_model.onnx') #Step 1: Load
# model.save(desired_model_path)

#tvmc.tune(model, target="llvm") #Step 1.5: Optional Tune #llvm

tvmc.compile(model, target="llvm -mcpu=cascadelake", package_path="whatever2") #Step 2: Compile

new_package = tvmc.TVMCPackage(package_path="whatever2")

result = tvmc.run(new_package, device="cpu") #Step 3: Run

print(result)