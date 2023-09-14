
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from ocr_package.model import User, Card



class RegistrationForm(FlaskForm):
    surname = StringField('Nom', validators=[DataRequired(), Length(min=2, max=20)])
    name = StringField('Prénom', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('E-mail', validators=[DataRequired(), Email()])
    phone = StringField('Numéro mobile', validators=[DataRequired(), Length(min=10)])
    password = PasswordField('Mot de passe', validators=[DataRequired()])
    confirm_password = PasswordField('Mot de passe confirmer', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField("S'inscrire")

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Cet e-mail est pris. Veuillez en choisir un autre.')


class LoginForm(FlaskForm):
    email = StringField('E-mail', validators=[DataRequired(), Email()])
    password = PasswordField('Mot de passe', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField("Se connecter")


class CardForm(FlaskForm):
    # image_file = StringField('Image_file', validators=[DataRequired()])
    card_holder = StringField('Card holder')
    card_number = StringField('Card number')
    valid_date = StringField('Valid date')
    submit = SubmitField("Envoyer")

    # def validate_card(self, card_number):
    #     card = Card.query.filter_by(card_number=card_number.data).first()
    #     if card:
    #         raise ValidationError('Cette carte deja existe. Veuillez en choisir un autre.')
