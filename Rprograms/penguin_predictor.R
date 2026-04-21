library(tidyverse)
library(caret)
library(neuralnet)
library(palmerpenguins)

datos = penguins %>% 
  drop_na(species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g)

datos = datos %>%
  mutate(
    Adelie    = ifelse(species == "Adelie", 1, 0),
    Gentoo    = ifelse(species == "Gentoo", 1, 0),
    Chinstrap = ifelse(species == "Chinstrap", 1, 0)
  )

datos = datos %>%
  mutate(across(c(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g), scale))

muestra = createDataPartition(datos$species, p = 0.8, list = FALSE)
train = datos[muestra, ]
test  = datos[-muestra, ]

red.neuronal = neuralnet(
  Adelie + Gentoo + Chinstrap ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g,
  data   = train,
  hidden = c(2, 3),
  linear.output = FALSE
)

plot(red.neuronal)

prediccion = predict(red.neuronal, test)

specie.decod = apply(prediccion, 1, which.max)

test$species.pred = recode(specie.decod,
                           "1" = "Adelie",
                           "2" = "Gentoo",
                           "3" = "Chinstrap"
)

table(test$species, test$species.pred)
