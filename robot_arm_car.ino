# Code reference from CrazyCurly
# Link: https://github.com/Crazycurly/gesture_MeArm/blob/main/arduino/main.ino

#include <Servo.h>

Servo servo[3];
int default_angle[3] = {90, 135, 90};

int motor_direction_A = 2;
int motor_speed_A = 5;
int motor_direction_B = 4;
int motor_speed_B = 6;
int speed = 50;
int total = 0;

void controlCar(int total);

void setup()
{
    Serial.begin(115200);
    servo[0].attach(11);
    servo[1].attach(10);
    servo[2].attach(9);

    for (size_t i = 0; i < 3; i++)
    {
        servo[i].write(default_angle[i]);
    }

    pinMode(motor_direction_A, OUTPUT);
    pinMode(motor_speed_A, OUTPUT);
    pinMode(motor_direction_B, OUTPUT);
    pinMode(motor_speed_B, OUTPUT);
}

byte angle[3];
byte pre_angle[3];
long t = millis();

void loop()
{
    if (Serial.available()) // Ensure at least 4 bytes are available for reading
    {
        Serial.readBytes(angle, 3);
        for (size_t i = 0; i < 3; i++)
        {
            if (angle[i] != pre_angle[i])
            {
                servo[i].write(angle[i]);
                pre_angle[i] = angle[i];
            }
        }
        t = millis();

        int total = Serial.parseInt();
        controlCar(total);
    }

    if (millis() - t > 1000)
    {
        for (size_t i = 0; i < 3; i++)
        {
            servo[i].write(default_angle[i]);
            pre_angle[i] = default_angle[i];
        }
    }
}

void controlCar(int total)
{
    if (total == 1)
    {
        digitalWrite(motor_direction_A, HIGH);
        analogWrite(motor_speed_A, speed);
        digitalWrite(motor_direction_B, LOW);
        analogWrite(motor_speed_B, speed);
    }
    else if (total == 4)
    {
        digitalWrite(motor_direction_A, LOW);
        analogWrite(motor_speed_A, speed);
        digitalWrite(motor_direction_B, HIGH);
        analogWrite(motor_speed_B, speed);
    }
    else if (total == 2)
    {
        digitalWrite(motor_direction_A, HIGH);
        analogWrite(motor_speed_A, speed);
        digitalWrite(motor_direction_B, HIGH);
        analogWrite(motor_speed_B, speed);
    }
    else if (total == 3)
    {
        digitalWrite(motor_direction_A, LOW);
        analogWrite(motor_speed_A, speed);
        digitalWrite(motor_direction_B, LOW);
        analogWrite(motor_speed_B, speed);
    }
    else if (total == 0 || total == 5)
    {
        digitalWrite(motor_direction_A, LOW);
        analogWrite(motor_speed_A, 0);
        digitalWrite(motor_direction_B, HIGH);
        analogWrite(motor_speed_B, 0);
    }
}
