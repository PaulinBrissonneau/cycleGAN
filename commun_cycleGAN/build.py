#@title Build and compile the model with its optimizers

from tensorflow.keras.optimizers import Adam

ALPHA = 0.0002 #@param {type:"number"}
BETA_1 = 0.5 #@param {type:"number"}

# optimizers
gen_A_to_B_optimizer = Adam(lr=ALPHA, beta_1=BETA_1)
gen_B_to_A_optimizer = Adam(lr=ALPHA, beta_1=BETA_1)
disc_A_optimizer = Adam(lr=ALPHA, beta_1=BETA_1)
disc_B_optimizer = Adam(lr=ALPHA, beta_1=BETA_1)

# model
model = cycleGAN(name=f'cycleGAN_{DATASET}',
                 disc_A = disc_A,
                 disc_B = disc_B,
                 gen_A_to_B = gen_A_to_B,
                 gen_B_to_A = gen_B_to_A,
                 disc_A_optimizer = disc_A_optimizer,
                 disc_B_optimizer = disc_B_optimizer,
                 gen_A_to_B_optimizer = gen_A_to_B_optimizer,
                 gen_B_to_A_optimizer = gen_B_to_A_optimizer,

                 )