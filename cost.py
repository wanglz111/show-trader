def martingale_total_cost(n=4, F1=1, P1=1000, m=1.6, trigger_percent=8):
    """
    计算前 n 次 martingale 加仓的累计总成本 S(n)

    n : int                第 n 层
    F1: float              第一次买入数量
    P1: float              第一次买入价格
    m : float              martingale 乘数
    trigger_percent: float 下跌触发百分比 (例如 10 表示跌 10%)
    """

    if n < 1:
        return 0.0
    if n == 1:
        return P1 * F1

    d = trigger_percent / 100.0
    r = 1 - d                      # 每次价格下降比例
    a = r * (m + 1)

    # a == 1 时，求和变成线性项（几乎不会发生，但为完整性处理）
    if abs(a - 1) < 1e-12:
        # S = P1*F1 * [1 + mr*(n-1)]
        return P1 * F1 * (1 + m * r * (n - 1))

    # 总成本 closed-form:
    S = P1 * F1 * (1 + m * r * (1 - a**(n - 1)) / (1 - a))
    return S

def base_position_size(trigger_mult, martingale_trigger):
    """
    计算基础仓位比例 F1%

    基础仓位是开仓+补仓3次后用完100%资金的仓位比例
    trigger_percent: float 下跌触发百分比 (例如 10 表示跌 10%)
    martingale_trigger: float martingale 触发百分比
    """
    martingale_total = martingale_total_cost(n=4, F1=1, P1=100, m=trigger_mult, trigger_percent=martingale_trigger)
    F1_pct = 100.0 / martingale_total
    return F1_pct


if __name__ == "__main__":
    # 示例计算
    n = 4
    F1 = 1
    P1 = 1000
    m = 1.2
    trigger_percent = 7

    total_cost = martingale_total_cost(n, F1, P1, m, trigger_percent)
    print(f"Martingale Total Cost S({n}) = {total_cost}")

    trigger_mult = 1.4
    martingale_trigger = 7
    base_pct = base_position_size(trigger_mult, martingale_trigger)
    print(f"Base Position Size F1% = {base_pct}%")