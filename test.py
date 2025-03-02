import torch
import sys
import os
import dancher_tools_segmentation as dt

# 确保当前路径在sys.path中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 解析参数
    args = dt.utils.get_config()
    # args.weights = 'UNet_best.pth'
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据加载器 (测试集)
    _, test_loader = dt.utils.get_dataloaders(args)

    # 初始化模型
    model = dt.utils.get_model(args, device)
    
    # 获取预定义指标
    metrics = dt.utils.get_metrics(args)


    model.load(model_dir=args.model_save_dir, mode=args.load_mode, specified_path=args.weights)

    # 定义损失函数
    criterion = dt.utils.get_loss(args)

    # 配置模型
    model.compile(optimizer=None, criterion=criterion, metrics=metrics, loss_weights=args.loss_weights)

    # 开始评估模型
    test_results = model.evaluate(data_loader=test_loader)


if __name__ == '__main__':
    main()
