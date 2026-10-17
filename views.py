from base import app, modelo
from flask import render_template, request


@app.route('/')
def homepage():
    return render_template('fundamentacao_teorica.html')


@app.route('/resultados')
def resultados():
    try:
        dados_res = modelo.obter_dados_resultados()
    except FileNotFoundError as erro:
        app.logger.error("Dataset não encontrado: %s", erro)
        return str(erro), 500
    return render_template(
        'resultados.html',
        stats=dados_res['stats'],
        metricas=dados_res['metricas']
    )


@app.route('/simulador', methods=['GET', 'POST'])
def simulador():
    contexto = {
        'form': request.form,
        'partidas': [],
        'times': [],
        'resultado': None,
        'erro': None,
    }
    status = 200
    try:
        contexto['partidas'] = modelo.lista_partidas()
        contexto['times'] = modelo.lista_times()
        if request.method == 'POST':
            contexto['resultado'] = modelo.simular(request.form)
    except ValueError as erro:          # entrada inválida do usuário
        contexto['erro'] = str(erro)
        status = 400
    except FileNotFoundError as erro:   # CSV ausente
        contexto['erro'] = str(erro)
        status = 500
    except Exception:                   # qualquer outra falha: registra e mostra mensagem amigável
        app.logger.exception("Falha inesperada no simulador")
        contexto['erro'] = "Erro inesperado ao simular. Tente novamente."
        status = 500
    return render_template('simulador.html', **contexto), status