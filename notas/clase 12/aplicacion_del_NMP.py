import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW   
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
# Cargar el dataset

# 1. DATOS DE EJEMPLO (comentarios en español)
comentarios = [
    # Positivos (1)
    "Me encantó este producto, es excelente",
    "Muy buen servicio, lo recomiendo totalmente",
    "La atención al cliente fue increíble",
    "Excelente calidad, superó mis expectativas",
    "Me siento muy satisfecho con mi compra",
    "El producto llegó antes de lo esperado",
    "Totalmente recomendado, funciona perfecto",
    "La mejor compra que he hecho este año",
    "El personal es muy amable y profesional",
    "Volvería a comprar sin dudarlo",
    
    # Negativos (0)
    "Pésimo servicio, no lo recomiendo",
    "El producto llegó roto y en mal estado",
    "Muy mala experiencia, no volveré a comprar",
    "La atención al cliente es terrible",
    "No funciona como esperaba, decepcionado",
    "El envío tardó mucho más de lo debido",
    "Producto de baja calidad, no vale lo que cuesta",
    "Me arrepiento de haber comprado esto",
    "El servicio al cliente no responde",
    "No cumple con lo prometido, muy malo"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Detectar si hay una GPU disponible y usarla, de lo contrario usar la CPU
#print(f"Usando dispositivo: {device}")

etiquetas = [1] * 10 + [0] * 10  # 1 para positivos, 0 para negativos
print(etiquetas)

#Create dataframe
df = pd.DataFrame({'comentarios': comentarios, 'etiquetas': etiquetas})
print(df)

#creamos una clase para personalizar nuestro conjunto de datos
class ComentariosDataset(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_length=128):
        self.textos = textos
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        texto = str(self.textos[idx])
        etiqueta = self.etiquetas[idx]
        encoding = self.tokenizer(
            texto,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_id': encoding['input_id'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(), #flatten convierte las listas anidadas en una sola lista
            'labels': torch.tensor(etiqueta, dtype=torch.long)
        }
    
# 3. PREPARAR LOS DATOS PARA EL ENTRENAMIENTO
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train, X_test, y_train, y_test = train_test_split(
    df['comentarios'].values,
    df['etiquetas'].values,
    test_size=0.3,
    random_state=23,
    stratify=df['etiquetas'].values,
)
print(f"Entrenamiento: {len(X_train)} muestras, Prueba: {len(X_test)} muestras")

train_dataset = ComentariosDataset(X_train, y_train, tokenizer)
test_dataset = ComentariosDataset(X_test, y_test, tokenizer)

print(f"Ejemplo de tokenización: {train_dataset}")

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

#Entrenamiento del modelo
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, )
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Entrenamiento
def entrenar_modelo(model, train_loader, test_loader, optimizer, epocas=5):
    train_losses = []
    test_accuracies = []
    for epoca in range(epocas):
        print(f"Época {epoca + 1}/{epocas}")
        print('='*50)
        # Modo entrenamiento
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Entrenando")):
            # Mover los datos al dispositivo
            input_id = batch['input_id'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            ## Pasar
            optimizer.zero_grad()
            outputs = model(
                input_id=input_id, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss
            num_batches += 1

            ##Pasar hacia atras

            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        ## Evaluazion
        accuracy = evaluar_modelo(model, test_loader)
        test_accuracies.append(accuracy)
        print(f"Pérdida de entrenamiento: {avg_train_loss:.4f}, Precision de prueba: {accuracy:.4f}")
    return train_losses, test_accuracies

def evaluar_modelo(model, test_loader):
    model.eval()
    predicciones = []
    verdaderos = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluando")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, preds = torch.max(outputs.logits, dim=1)
                predicciones.extend(preds.cpu().tolist())
                verdaderos.extend(labels.cpu().tolist())
            except:
                print(f'Error evaluando {batch_idx}:')
                continue
    
    accuracy = accuracy_score(verdaderos, predicciones)
    return accuracy

# Entrenar el modelo
train_losses, test_accuracies = entrenar_modelo(
    model, train_dataloader, test_dataloader, optimizer, epocas=5
)
