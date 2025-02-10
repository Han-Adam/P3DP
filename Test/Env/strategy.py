import numpy as np
from .util import feet2meter, rad2degree


class Strategy:
    def __init__(self):
        self.height_protection = True
        self.escape = True

    def reset(self, strategy_num):
        self.height_protection = strategy_num[0]
        self.escape = strategy_num[1]

    def process(self, self_fighter, target_fighter):
        """
        1: straight fly
        2: climb
        3: lopping
        4: split_s
        5: attitude_tracking
        6: position_tracking
        7: high yoyo
        8: low yoyo
        """
        if self.height_protection:
            if self_fighter.height * feet2meter < 1000:
                return 2  # climb to increase height

        if self.escape:
            # line of sight, LOS
            los = (target_fighter.position - self_fighter.position) * feet2meter
            d = (los[0] ** 2 + los[1] ** 2 + los[2] ** 2) ** 0.5
            # antenna-train angle ATA
            ata = rad2degree * np.arccos(np.inner(los, self_fighter.heading) / d)
            # target-aspect angle TAA
            aa = rad2degree * np.arccos(np.inner(los, target_fighter.heading) / d)
            if d < 3000 and ata > 120 and aa > 120:
                if self_fighter.height > target_fighter.height:
                    return 3  # lopping, escape from high position
                else:
                    return 4  # split-s, escape from low position

        return 5  # attitude tracking
