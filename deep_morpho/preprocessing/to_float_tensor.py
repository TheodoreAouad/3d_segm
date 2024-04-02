from torchvision.transforms import ToTensor


class ToFloatTensor(ToTensor):
    def __call__(self, pic):
        img = super().__call__(pic)
        return img.float()
