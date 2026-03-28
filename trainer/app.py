from flask import Flask

from trainer import db
from trainer.routes import annotate, api, projects


def create_app() -> Flask:
    app = Flask(__name__)
    db.init_db()
    app.register_blueprint(projects.bp)
    app.register_blueprint(annotate.bp)
    app.register_blueprint(api.bp)
    return app


app = create_app()

