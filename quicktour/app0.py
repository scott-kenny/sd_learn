from diffusers import StableDiffusionPipeline, DiffusionPipeline

# 网络加载，并存储在本地
#pipeline = StableDiffusionPipeline.from_pretrained("windwhinny/chilloutmix")
#pipeline.save_pretrained("./models/chilloutmix/")

# 本地加载
pipeline = StableDiffusionPipeline.from_pretrained(
    "models/chilloutmix/",
    safety_checker = None,                   # 关闭安全检查
    requires_safety_checker = False          # 关闭安全检查
)
# 启动GPU运算
pipeline.to("cuda")



# 推理
prompt = "1girl, nude"
image = pipeline(prompt)[0][0]
image.save("outputs/start1.png")