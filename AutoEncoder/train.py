import numpy as np
import torch
from tqdm import tqdm
from model import Encoder, Decoder
from plot import plot_ae_outputs_den
from dataset import Dataset
from add_noise import add_noise


def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.3):
	# Set train mode for both the encoder and the decoder
	encoder.train()
	decoder.train()
	train_loss = []
	

	# Iterate the dataloader (we do not need the label values, this is unsupervised learning)
	for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
		# Move tensor to the proper device
		image_noisy = add_noise(image_batch,noise_factor)
		image_batch = image_batch.to(device)
		image_noisy = image_noisy.to(device)
		# Encode data
		encoded_data = encoder(image_noisy)
		# Decode data
		decoded_data = decoder(encoded_data)
		# Evaluate loss
		loss = loss_fn(decoded_data, image_batch)
		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# Print batch loss
		#print('\t partial train loss (single batch): %f' % (loss.data))
		train_loss.append(loss.detach().cpu().numpy())

	return np.mean(train_loss)

def test_epoch_den(encoder, decoder, device, dataloader, loss_fn,noise_factor=0.3):
	# Set evaluation mode for encoder and decoder
	encoder.eval()
	decoder.eval()
	with torch.no_grad(): # No need to track the gradients
		# Define the lists to store the outputs for each batch
		conc_out = []
		conc_label = []
		for image_batch, _ in dataloader:
			# Move tensor to the proper device
			image_noisy = add_noise(image_batch,noise_factor)
			image_noisy = image_noisy.to(device)
			# Encode data
			encoded_data = encoder(image_noisy)
			# Decode data
			decoded_data = decoder(encoded_data)
			# Append the network output and the original image to the lists
			conc_out.append(decoded_data.cpu())
			conc_label.append(image_batch.cpu())
		# Create a single tensor with all the values in the lists
		conc_out = torch.cat(conc_out)
		conc_label = torch.cat(conc_label)
		# Evaluate global loss
		val_loss = loss_fn(conc_out, conc_label)
	return val_loss.data

def train():
	### Load the dataset
    train_loader, valid_loader, test_loader = Dataset().getDataLoader()

	### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Define an optimizer (both for the encoder and the decoder!)
    lr= 0.001

    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    d = 10

    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
    print(optim)
    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)
	
    noise_factor = 0.3
    num_epochs = 10
    history_da={'train_loss':[],'val_loss':[]}

    for epoch in tqdm(range(num_epochs)):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training (use the training function)
        train_loss=train_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim,
            noise_factor=noise_factor)
        ### Validation (use the testing function)
        val_loss = test_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=valid_loader,
            loss_fn=loss_fn,noise_factor=noise_factor)
        # Print Validationloss
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        plot_ae_outputs_den(encoder, decoder, epoch=epoch, noise_factor=noise_factor)
	
    torch.save(encoder.state_dict(), 'results/encoder.pth')
    torch.save(decoder.state_dict(), 'results/decoder.pth')
    print('Model saved!')


if __name__ == '__main__':
	train()