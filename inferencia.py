from pathlib import Path
import pickle

import pandas as pd


ARQUIVO_MODELO = Path(__file__).resolve().parent / "breast_cancer_model.pkl"
OPCOES = {
    "age": ("10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"),
    "menopause": ("lt40", "ge40", "premeno"),
    "tumor-size": (
        "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
        "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
    ),
    "inv-nodes": (
        "0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20",
        "21-23", "24-26", "27-29", "30-32", "33-35", "36-39",
    ),
    "node-caps": ("yes", "no"),
    "deg-malig": ("1", "2", "3"),
    "breast": ("left", "right"),
    "breast-quad": ("left_up", "left_low", "right_up", "right_low", "central"),
    "irradiat": ("yes", "no"),
}


def prever_paciente(nova_paciente):
    with ARQUIVO_MODELO.open("rb") as arquivo:
        artefato = pickle.load(arquivo)

    atributos = artefato["atributos"]
    faltantes = [atributo for atributo in atributos if atributo not in nova_paciente]
    if faltantes:
        raise ValueError("Atributos nao informados: " + ", ".join(faltantes))

    valores = {}
    for atributo in atributos:
        valor = nova_paciente[atributo]
        valor = str(valor).strip() if valor is not None else ""
        if valor and valor not in OPCOES[atributo]:
            raise ValueError(
                f"Valor invalido para {atributo}: {valor}. "
                f"Opcoes: {', '.join(OPCOES[atributo])}"
            )
        valores[atributo] = valor or None

    paciente = pd.DataFrame(
        [valores],
        columns=atributos,
    )
    return artefato["modelo"].predict(paciente)[0]


def main():
    paciente = {}
    print("Informe os dados da paciente. Deixe em branco se nao souber um valor.")
    for atributo, opcoes in OPCOES.items():
        valor = input(f"{atributo} ({', '.join(opcoes)}): ").strip()
        paciente[atributo] = valor if valor else None

    classe = prever_paciente(paciente)
    if classe == "recurrence-events":
        print("Indicativo previsto: recurrence-events (recorrencia)")
    else:
        print("Indicativo previsto: no-recurrence-events (sem recorrencia)")
    print("Este resultado e apenas uma classificacao experimental, nao um diagnostico medico.")


if __name__ == "__main__":
    main()
