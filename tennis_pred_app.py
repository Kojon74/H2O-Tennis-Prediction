from h2o_wave import Q, main, app, ui
from tensorflow import keras
import numpy as np
import json
import os


def load_ids():
    id_jsons = sorted(os.listdir("ids"))
    id_dicts = []
    for id_json in id_jsons:
        with open(os.path.join("ids", id_json), "r") as f_json:
            id_dict = json.load(f_json)
            id_dicts.append(id_dict)
    return id_dicts


player_ids, tourn_ids = load_ids()
round_id = {
    "RR": 1,
    "BR": 2,
    "R128": 3,
    "R64": 4,
    "R32": 5,
    "R16": 6,
    "QF": 7,
    "SF": 8,
    "F": 9,
}
round_list = ["RR", "BR", "R128", "R64", "R32", "R16", "QF", "SF", "F"]


def on_startup():
    pass


@app("/tennis-pred", on_startup=on_startup)
async def serve(q: Q):
    if q.args.submit:
        calc_winner(q)
    else:
        show_form(q)
    await q.page.save()


def show_form(q: Q):
    q.page["meta"] = ui.meta_card(
        box="",
        title="Tennis Match Prediction",
        refresh=1,
        layouts=[
            ui.layout(
                breakpoint="xs",
                width="1200px",
                zones=[
                    ui.zone("header"),
                    ui.zone("tournament"),
                    ui.zone(
                        "players",
                        direction=ui.ZoneDirection.ROW,
                        zones=[
                            ui.zone("player1", size="50%"),
                            ui.zone("player2", size="50%"),
                        ],
                    ),
                    ui.zone("submit", size="50%"),
                    ui.zone("result", size="50%"),
                ],
            )
        ],
    )

    q.page["header"] = ui.header_card(
        box="header", title="Tennis Match Prediction", subtitle=""
    )

    q.page["tournament"] = ui.form_card(
        box="tournament",
        items=[
            # Tournament name
            ui.dropdown(
                name="t_name",
                label="Tournament Name",
                placeholder="Wimbledon",
                choices=[
                    ui.choice(name=t_name, label=t_name) for t_name in tourn_ids.keys()
                ],
                trigger=False,
            ),
            # Round
            ui.dropdown(
                name="t_round",
                label="Round",
                placeholder="Final",
                choices=[
                    ui.choice(name=t_round, label=t_round) for t_round in round_list
                ],
                trigger=False,
            ),
        ],
    )

    q.page["p1"] = ui.form_card(
        box="player1",
        items=[
            ui.text_l("Player 1"),
            ui.dropdown(
                name="p1_name",
                label="Name",
                placeholder="Roger Federer",
                choices=[
                    ui.choice(name=p_name, label=p_name) for p_name in player_ids.keys()
                ],
                trigger=False,
            ),
            ui.textbox(name="p1_rank", label="Rank"),
            ui.textbox(name="p1_age", label="Age"),
        ],
    )

    q.page["player2"] = ui.form_card(
        box="player2",
        items=[
            ui.text_l("Player 2"),
            ui.dropdown(
                name="p2_name",
                label="Name",
                placeholder="Rafael Nadal",
                choices=[
                    ui.choice(name=p_name, label=p_name) for p_name in player_ids.keys()
                ],
                trigger=False,
            ),
            ui.textbox(name="p2_rank", label="Rank"),
            ui.textbox(name="p2_age", label="Age"),
        ],
    )

    q.page["submit"] = ui.form_card(
        box="submit",
        items=[
            ui.buttons(
                items=[ui.button(name="submit", label="Submit", primary=True)],
                justify="center",
            )
        ],
    )

    result_str = "" if not q.client.result_str else q.client.result_str

    q.page["result"] = ui.markdown_card(
        box="result",
        title="Results",
        content=result_str,
    )


def calc_winner(q: Q):
    t_name = q.args.t_name
    t_round = q.args.t_round
    p1_name = q.args.p1_name
    p1_rank = int(q.args.p1_rank)
    p1_age = int(q.args.p1_age)
    p2_name = q.args.p2_name
    p2_rank = int(q.args.p2_rank)
    p2_age = int(q.args.p2_age)

    t_round_id = round_id[t_round]

    data1 = [
        tourn_ids[t_name]
        + [t_round_id]
        + player_ids[p1_name]
        + [p1_age, p1_rank]
        + player_ids[p2_name]
        + [p2_age, p2_rank]
    ]
    data2 = [
        tourn_ids[t_name]
        + [t_round_id]
        + player_ids[p2_name]
        + [p2_age, p2_rank]
        + player_ids[p1_name]
        + [p1_age, p1_rank]
    ]

    # print(
    #     f"t_name: {t_name}, p1_name: {p1_name}, p1_rank: {p1_rank}, p1_age: {p1_age}, p2_name: {p2_name}, p2_rank: {p2_rank}, p2_age: {p2_age}"
    # )

    model = keras.models.load_model("models")
    pred1 = model.predict(data1)
    pred2 = model.predict(data2)
    prediction = np.asarray([(pred1[0][i] + pred2[0][1 - i]) / 2 for i in range(2)])
    players = [p1_name, p2_name]
    winner = players[prediction.argmax()]
    probability = np.amax(prediction) * 100

    q.client.result_str = (
        f"Predicted winner is: {winner} with probability: {probability}%"
    )

    show_form(q)
