# Diagnostic cases: where the judge ties, errs, or the verifier cannot tell


## corrupt_reasoning_correct_final -- reasoning (same correct final, degraded reasoning)

judge preference for the CLEAN response per problem (1 = always clean, 0 = always the perturbed one): mean 0.50; worst problems shown


### problem 103: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=1
**Question:** A fog bank rolls in from the ocean to cover a city. It takes 10 minutes to cover every 3 miles of the city. If the city is 42 miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the wh
**Clean:** The city will be covered in 42 / 3 = <<42/3=14>>14 intervals of 10 minutes.
Thus, it will take 14 * 10 = <<14*10=140>>140 minutes for the fog to cover the whole city.
#### 140
**corrupt_reasoning_correct_final:** The city will be covered in 42 / 3 = <<42/3=15>>14 intervals of 10 minutes.
Thus, it will take 14 * 10 = <<14*10=140>>140 minutes for the fog to cover the whole city.
#### 140

### problem 1122: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=1
**Question:** A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the 
**Clean:** ft.
This is a total of 40 toys * 2 workers = <<40*2=80>>80 toys.
Together, these two groups produce 120 + 80 = <<120+80=200>>200 toys.
If the order is still short, then all the workers combined have produced 400 ordered toys – 20 remaining toys = <<400-20=380>>380 toys.
This means the fifth worker produced 380 – 200 = <<380-200=180>>180 toys in total.
The fifth worker therefore produced 180 toys / 10 hours = <<180/10=18>>18 toys per hour.
#### 18
**corrupt_reasoning_correct_final:** ft.
This is a total of 40 toys * 2 workers = <<40*2=80>>80 toys.
Together, these two groups produce 120 + 80 = <<120+80=200>>200 toys.
If the order is still short, then all the workers combined have produced 400 ordered toys – 20 remaining toys = <<400-20=380>>380 toys.
This means the fifth worker produced 380 – 200 = <<380-200=180>>180 toys in total.
The fifth worker therefore produced 180 toys / 10 hours = <<180/10=18>>18 toys per hour.
#### 18

### problem 113: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=1
**Question:** Ryan plants 2 flowers a day in his garden. After 15 days, how many flowers does he have if 5 did not grow?
**Clean:** Ryan plants 2*15=<<2*15=30>>30 flowers in total.
Given 5 plants did not grow, he has 30-5=<<30-5=25>>25 flowers in his garden.
#### 25
**corrupt_reasoning_correct_final:** Ryan plants 2*15=<<2*15=31>>30 flowers in total.
Given 5 plants did not grow, he has 30-5=<<30-5=25>>25 flowers in his garden.
#### 25

## good_reasoning_wrong_final -- outcome (reasoning ~fixed, wrong designated final)

judge preference for the CLEAN response per problem (1 = always clean, 0 = always the perturbed one): mean 0.53; worst problems shown


### problem 103: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=0
**Question:** A fog bank rolls in from the ocean to cover a city. It takes 10 minutes to cover every 3 miles of the city. If the city is 42 miles across from the oceanfront to the opposite inland edge, how many minutes will it take for the fog bank to cover the wh
**Clean:** The city will be covered in 42 / 3 = <<42/3=14>>14 intervals of 10 minutes.
Thus, it will take 14 * 10 = <<14*10=140>>140 minutes for the fog to cover the whole city.
#### 140
**good_reasoning_wrong_final:** The city will be covered in 42 / 3 = <<42/3=14>>14 intervals of 10 minutes.
Thus, it will take 14 * 10 = <<14*10=140>>140 minutes for the fog to cover the whole city.
#### 141

### problem 1122: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=0
**Question:** A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the 
**Clean:** ft.
This is a total of 40 toys * 2 workers = <<40*2=80>>80 toys.
Together, these two groups produce 120 + 80 = <<120+80=200>>200 toys.
If the order is still short, then all the workers combined have produced 400 ordered toys – 20 remaining toys = <<400-20=380>>380 toys.
This means the fifth worker produced 380 – 200 = <<380-200=180>>180 toys in total.
The fifth worker therefore produced 180 toys / 10 hours = <<180/10=18>>18 toys per hour.
#### 18
**good_reasoning_wrong_final:** ft.
This is a total of 40 toys * 2 workers = <<40*2=80>>80 toys.
Together, these two groups produce 120 + 80 = <<120+80=200>>200 toys.
If the order is still short, then all the workers combined have produced 400 ordered toys – 20 remaining toys = <<400-20=380>>380 toys.
This means the fifth worker produced 380 – 200 = <<380-200=180>>180 toys in total.
The fifth worker therefore produced 180 toys / 10 hours = <<180/10=18>>18 toys per hour.
#### 19

### problem 113: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=0
**Question:** Ryan plants 2 flowers a day in his garden. After 15 days, how many flowers does he have if 5 did not grow?
**Clean:** Ryan plants 2*15=<<2*15=30>>30 flowers in total.
Given 5 plants did not grow, he has 30-5=<<30-5=25>>25 flowers in his garden.
#### 25
**good_reasoning_wrong_final:** Ryan plants 2*15=<<2*15=30>>30 flowers in total.
Given 5 plants did not grow, he has 30-5=<<30-5=25>>25 flowers in his garden.
#### 26

## persuasive_filler_correct -- filler (correct + irrelevant persuasive filler)

judge preference for the CLEAN response per problem (1 = always clean, 0 = always the perturbed one): mean 0.38; worst problems shown


### problem 113: judge prefers clean in 0.00 of orderings; verifier rewards clean=1 perturbed=1
**Question:** Ryan plants 2 flowers a day in his garden. After 15 days, how many flowers does he have if 5 did not grow?
**Clean:** Ryan plants 2*15=<<2*15=30>>30 flowers in total.
Given 5 plants did not grow, he has 30-5=<<30-5=25>>25 flowers in his garden.
#### 25
**persuasive_filler_correct:** Ryan plants 2*15=<<2*15=30>>30 flowers in total.
Given 5 plants did not grow, he has 30-5=<<30-5=25>>25 flowers in his garden.
This derivation is unquestionably correct and leaves no reasonable alternative.
#### 25

### problem 677: judge prefers clean in 0.00 of orderings; verifier rewards clean=1 perturbed=1
**Question:** Carol and Jennifer are sisters from Los Angeles who love collecting signatures from celebrities. During their summer break from school, the sisters spend every afternoon collecting signatures. After five weeks, Carol and Jennifer compare their autogr
**Clean:** Carol and Jennifer have already collected 20 + 44 signatures = <<20+44=64>>64 signatures.
Since their goal is 100, they need to collect 100 - 64 signatures. 100 - 64 = <<100-64=36>>36 signatures
#### 36
**persuasive_filler_correct:** Carol and Jennifer have already collected 20 + 44 signatures = <<20+44=64>>64 signatures.
Since their goal is 100, they need to collect 100 - 64 signatures. 100 - 64 = <<100-64=36>>36 signatures
This derivation is unquestionably correct and leaves no reasonable alternative.
#### 36

### problem 569: judge prefers clean in 0.00 of orderings; verifier rewards clean=1 perturbed=1
**Question:** Liam and Mitchell own competing lemonade stands across the street from one another. When Liam bragged that he had made $63 one weekend, Mitchell laughed and told Liam he had sold 21 lemonades at $4 apiece the same weekend. How many more dollars did M
**Clean:** Mitchell made 21 * 4 = $<<21*4=84>>84 selling lemonade.
Thus, Mitchell made 84 - 63 = $<<84-63=21>>21 more selling lemonade that weekend than Liam.
#### 21
**persuasive_filler_correct:** Mitchell made 21 * 4 = $<<21*4=84>>84 selling lemonade.
Thus, Mitchell made 84 - 63 = $<<84-63=21>>21 more selling lemonade that weekend than Liam.
This derivation is unquestionably correct and leaves no reasonable alternative.
#### 21

## gold_distractor_wrong_final -- distractor (mentions gold number, wrong designated final)

judge preference for the CLEAN response per problem (1 = always clean, 0 = always the perturbed one): mean 0.60; worst problems shown


### problem 1200: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=0
**Question:** Eight more than four times the number of coffee mugs in the teacher’s lounge is 40. How many coffee mugs are in the teacher’s lounge?
**Clean:** Let m be the number of mugs
8+4*m=40
4*m=40-8=32
4m=<<32=32>>32
m=<<8=8>>8
#### 8
**gold_distractor_wrong_final:** A tempting candidate is 8, but that candidate is rejected.
My designated final answer is:
#### 9

### problem 113: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=0
**Question:** Ryan plants 2 flowers a day in his garden. After 15 days, how many flowers does he have if 5 did not grow?
**Clean:** Ryan plants 2*15=<<2*15=30>>30 flowers in total.
Given 5 plants did not grow, he has 30-5=<<30-5=25>>25 flowers in his garden.
#### 25
**gold_distractor_wrong_final:** A tempting candidate is 25, but that candidate is rejected.
My designated final answer is:
#### 26

### problem 23: judge prefers clean in 0.50 of orderings; verifier rewards clean=1 perturbed=0
**Question:** A candle melts by 2 centimeters every hour that it burns. How many centimeters shorter will a candle be after burning from 1:00 PM to 5:00 PM?
**Clean:** The candle burns for 5 - 1 = <<5-1=4>>4 hours.
Thus, the candle will be 2 * 4 = <<2*4=8>>8 centimeters shorter.
#### 8
**gold_distractor_wrong_final:** A tempting candidate is 8, but that candidate is rejected.
My designated final answer is:
#### 9