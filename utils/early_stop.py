"""Early stopping — BA가 patience회 연속 평가에서 개선되지 않으면 학습 중단."""


class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_ba   = -1.0
        self.counter   = 0

    def step(self, ba: float) -> bool:
        """BA를 기록. patience회 연속 개선이 없으면 True(중단 신호) 반환."""
        if ba > self.best_ba:
            self.best_ba = ba
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
