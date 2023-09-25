
class Monster:
    def __init__(self,health, energy):
        self.health = health
        self.energy = energy

    def get_damage(self, amount):
        self.health -= amount

    def move(self, speed):
        print(f'it has a speed of {speed}')

class Scorpion(Monster):
    def __init__(self, poison_damage, health, energy):
        super().__init__(health, energy)
        self.poison_damage = poison_damage

    def get_demage(self):
        print(f'poison damage is {self.poison_damage}')