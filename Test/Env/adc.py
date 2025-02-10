from .util import rho0


def adc(groundSpeed, height):
    '''air data computation'''
    tfac = 1 - .703e-5 * height
    temperature = 390 if height >= 35000 else 519 * tfac
    rho = rho0 * tfac ** 4.14  # air mass density
    soundSpeed = (1.4 * 1716.3 * temperature) ** 0.5  # sqrt(adiabatic exponent * gas constant * temperature)
    mach = groundSpeed / soundSpeed
    qbar = rho * groundSpeed ** 2 / 2  # dynamic pressure
    # ps = 1715 * rho * temperature # static pressure
    return mach, qbar