from model import *
from squad_data import *


class Trainer:
    def __init__(self, config):
        self.BATCH_SIZE = config.BATCH_SIZE
        self.ENC_EMB_DIM = config.ENC_EMB_DIM
        self.DEC_EMB_DIM = config.DEC_EMB_DIM
        self.HID_DIM = config.HID_DIM
        self.N_LAYERS = config.N_LAYERS
        self.ENC_DROPOUT = config.ENC_DROPOUT
        self.DEC_DROPOUT = config.DEC_DROPOUT
        self.N_EPOCHS = config.N_EPOCHS
        self.CLIP = config.CLIP
        self.json_file = config.json_file
        self.json_test_file = config.json_test_file
        self.load_model = config.load_model

        SquadSourceField = Field(
            tokenize=tokenize, init_token="<sos>", eos_token="<eos>", lower=True
        )
        train_data = SquadDataset(self.json_file, SquadSourceField)
        test_data = SquadDataset(self.json_test_file, SquadSourceField)

        SquadSourceField.build_vocab(
            train_data, vectors=GloVe(name="6B", dim=50), min_freq=2
        )
        print(f"Unique tokens in Source vocabulary: {len(SquadSourceField.vocab)}")

        self.INPUT_DIM = len(SquadSourceField.vocab)
        self.OUTPUT_DIM = len(SquadSourceField.vocab)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        (
            self.train_iterator,
            self.valid_iterator,
            self.test_iterator,
        ) = BucketIterator.splits(
            (train_data, test_data, test_data),
            batch_size=self.BATCH_SIZE,
            device=self.device,
        )

        enc = Encoder(
            self.INPUT_DIM,
            self.ENC_EMB_DIM,
            self.HID_DIM,
            self.N_LAYERS,
            self.ENC_DROPOUT,
        )
        dec = Decoder(
            self.OUTPUT_DIM,
            self.DEC_EMB_DIM,
            self.HID_DIM,
            self.N_LAYERS,
            self.DEC_DROPOUT,
        )

        model = Seq2Seq(enc, dec, self.device).to(self.device)
        model.apply(init_weights)
        print(f"The model has {count_parameters(model):,} trainable parameters")

        self.optimizer = optim.Adam(model.parameters())

        self.criterion = nn.CrossEntropyLoss()

        self.model = model

    def get_model_details(self):
        return (
            self.device,
            self.train_iterator,
            self.valid_iterator,
            self.test_iterator,
            self.model,
            self.optimizer,
            self.criterion,
        )

    def ModelFitter(self):

        (
            device,
            train_iterator,
            valid_iterator,
            test_iterator,
            model,
            optimizer,
            criterion,
        ) = self.get_model_details()
        best_valid_loss = float("inf")

        for epoch in range(self.N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion, self.CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins = int((end_time - start_time) / 60)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), self.load_model)

            # valid_loss_em = evaluate(model, valid_iterator, "em")
            # valid_loss_f1 = evaluate(model, valid_iterator, "f1")
            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins} minutes")
            print(f"\tTrain Loss: {train_loss:.3f} ")
            print(f"\t Val. Loss: {valid_loss:.3f} ")
            # print(f"\t Val. EM: {valid_loss_em:.3f} ")
            # print(f"\t Val. F1: {valid_loss_f1:.3f} ")

    def ModelScorer(self):
        (
            device,
            train_iterator,
            valid_iterator,
            test_iterator,
            model,
            optimizer,
            criterion,
        ) = self.get_model_details()

        model.load_state_dict(torch.load(self.load_model))

        test_loss = evaluate(model, test_iterator, criterion)

        print(f"Test Loss: {test_loss:.3f} ")


def main(config):

    # Initialize the model
    t = Trainer(config)

    # Fit on the train data and evaluate using the Validation data
    t.ModelFitter()

    # Score on the test data (Same as validation data for this exercise)
    t.ModelScorer()


if __name__ == "__main__":
    main(config)

