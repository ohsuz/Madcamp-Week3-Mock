package com.app.madcampweek3.ui.capture;

import android.Manifest;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.EditText;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;
import androidx.fragment.app.Fragment;
import androidx.loader.content.CursorLoader;

import com.app.madcampweek3.CaptureAdapter;
import com.app.madcampweek3.CaptureItem;
import com.app.madcampweek3.PopupActivity;
import com.app.madcampweek3.R;
import com.app.madcampweek3.User;
import com.app.madcampweek3.api.RetroApi;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.UUID;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import static android.app.Activity.RESULT_OK;

public class CaptureFragment extends Fragment {

    // Retrofit http communication api setting
    private String BASE_URL = "http://192.249.19.242:6080/";
    private String TAG = "TAG";

    private Retrofit retrofit = new Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build();

    private RetroApi retroApi = retrofit.create(RetroApi.class);
    private String tempAnswer = "";

    /*
    서버에 보낼 정보: 학년, 년도, 월, 과목 + 문제 번호
     */
    String grade, year, month, subject;

    CaptureAdapter adapter;
    GridView gridView;

    Button correctBtn;
    Button imageBtn;

    Spinner spinner_year, spinner_month, spinner_subject;
    RadioButton first, second, third;
    RadioGroup radioGroup;

    ArrayList<String> question = new ArrayList<String>();
    HashMap<String, String> qanda = new HashMap<String, String>();
    HashMap<String, String> myAnswer = new HashMap<String, String>();
    int wrongNumber = 0;
    String mock_id = "";

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        View root = inflater.inflate(R.layout.fragment_capture, container, false);
        ActivityCompat.requestPermissions(getActivity(), new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1000);
        ActivityCompat.requestPermissions(getActivity(), new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1000);

        /*
        RADIO GROUP
         */
        radioGroup = (RadioGroup) root.findViewById(R.id.radioGroup);
        first = (RadioButton) root.findViewById(R.id.first);
        second = (RadioButton) root.findViewById(R.id.second);
        third = (RadioButton) root.findViewById(R.id.third);

        radioGroup.setOnCheckedChangeListener(radioGroupButtonChangeListener);

        /*
        SPINNER
         */
        spinner_year = (Spinner) root.findViewById(R.id.year);
        spinner_month = (Spinner) root.findViewById(R.id.month);
        spinner_subject = (Spinner) root.findViewById(R.id.subject);
        spinner_year.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                year = (String) adapterView.getItemAtPosition(i);
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
            }
        });
        spinner_month.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                month = (String) adapterView.getItemAtPosition(i);
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
            }
        });
        spinner_subject.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                subject = (String) adapterView.getItemAtPosition(i);
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {
            }
        });

        gridView = root.findViewById(R.id.gridView);
        adapter = new CaptureAdapter();

        imageBtn = (Button) root.findViewById(R.id.imageBtn);
        correctBtn = (Button) root.findViewById(R.id.correctBtn);

        imageBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //  EXTRA_ALLOW_MULTIPLE: 갤러리에서 다중 선택이 가능하도록 함
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI);
                //intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                intent.setType("image/*");
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), 10);
            }
        });

        correctBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                question.clear();
                qanda.clear();
                myAnswer.clear();
                wrongNumber = 0;

                /*
                OCR을 이용하여 문제 번호와 답(주관식의 경우)을 구해서 변수에 저장함
                 */
                for (int i = 0; i < adapter.getCount(); i++) {
                    String imgPath = adapter.getItem(i).getImgPath();
                    NCP ncpThread = new NCP(imgPath);
                    ncpThread.start();
                    try {
                        ncpThread.join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                /*
                위에서 저장된 문제들의 정답을 서버에서 구해옴
                 */
                for (int i = 0; i < question.size(); i++) {
                    String thisQuestion = question.get(i);
                    GetAnswer getAnswer = new GetAnswer(thisQuestion);
                    getAnswer.start();
                    try {
                        getAnswer.join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                /*
                틀린 문제를 DB에 넣는 과정
                 */
                for (int i = 0; i < question.size(); i++) {
                    Log.d("aaaaaaa", "aaaaaaaaaaaRealAnswer" + qanda.get(question.get(i)));
                    Log.d("aaaaaaa", "aaaaaaaaaaaMyAnswer" + myAnswer.get(question.get(i)));
                    if (!qanda.get(question.get(i)).equals(myAnswer.get(question.get(i)))) {
                        Log.d("aaaaaaa", "aaaaaaaaaa" + question.get(i) + "번 문제 틀림");
                        wrongNumber++;
                        InsertWrong insertWrong = new InsertWrong(question.get(i));
                        insertWrong.start();
                        try {
                            insertWrong.join();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }

                GetMockId getMockId = new GetMockId();
                getMockId.start();
                try {
                    getMockId.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }


                // 전체 문제 수 - 틀린 문제 수 보여주고 바로 오답노트로 갈지 안 갈지
                Intent intent = new Intent(getContext(), PopupActivity.class);
                intent.putExtra("wrong", wrongNumber);
                intent.putExtra("total", question.size());
                intent.putExtra("mock_id", mock_id);
                startActivity(intent);
            }
        });

        return root;
    }

    RadioGroup.OnCheckedChangeListener radioGroupButtonChangeListener = new RadioGroup.OnCheckedChangeListener() {
        @Override
        public void onCheckedChanged(RadioGroup radioGroup, int i) {
            if (i == R.id.first) {
                grade = "1";
            } else if (i == R.id.second) {
                grade = "2";
            } else {
                grade = "3";
            }
        }
    };

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 10 && resultCode == RESULT_OK) {
            Uri imgUri = data.getData();
            String imgPath = getRealPathFromUri(imgUri);

            adapter.addItem(new CaptureItem(imgUri, imgPath));
            gridView.setAdapter(adapter);
        } else {
            Toast.makeText(getContext(), "ERROR", Toast.LENGTH_SHORT).show();
        }
    }


    // Uri를 절대경로로 바꿔줌
    String getRealPathFromUri(Uri uri) {
        String[] proj = {MediaStore.Images.Media.DATA};
        CursorLoader loader = new CursorLoader(getActivity(), uri, proj, null, null, null);
        Cursor cursor = loader.loadInBackground();
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        String result = cursor.getString(column_index);
        cursor.close();
        return result;
    }


    public class NCP extends Thread {
        private String imgPath;

        public NCP(String imgPath) {
            this.imgPath = imgPath;
        }

        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        public void run() {
            String apiURL = "https://7fddcdc659404aaf911921f7567947c5.apigw.ntruss.com/custom/v1/2966/851445119e04578b0ad435fdf1f1d6812329d5720dcb40731ba7d698d6b158fa/general";
            String secretKey = "cVdHTUh0eUdPWG9QYXNiQXB4V0FhT1dYZ0RnQktnZnQ=";
            String imageFile = imgPath;
            try {
                URL url = new URL(apiURL);
                HttpURLConnection con = (HttpURLConnection) url.openConnection();
                con.setUseCaches(false);
                con.setDoInput(true);
                con.setDoOutput(true);
                con.setReadTimeout(30000);
                con.setRequestMethod("POST");
                String boundary = "----" + UUID.randomUUID().toString().replaceAll("-", "");
                con.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
                con.setRequestProperty("X-OCR-SECRET", secretKey);
                JSONObject json = new JSONObject();
                json.put("version", "V2");
                json.put("requestId", UUID.randomUUID().toString());
                json.put("timestamp", System.currentTimeMillis());
                JSONObject image = new JSONObject();
                image.put("format", "jpg");
                image.put("name", "demo");
                JSONArray images = new JSONArray();
                images.put(image);
                json.put("images", images);
                String postParams = json.toString();
                con.connect();
                DataOutputStream wr = new DataOutputStream(con.getOutputStream());
                long start = System.currentTimeMillis();
                File file = new File(imageFile);
                writeMultiPart(wr, postParams, file, boundary);
                wr.close();
                int responseCode = con.getResponseCode();
                BufferedReader br;
                if (responseCode == 200) {
                    br = new BufferedReader(new InputStreamReader(con.getInputStream()));
                } else {
                    br = new BufferedReader(new InputStreamReader(con.getErrorStream()));
                }
                String inputLine = br.readLine();
                try {
                    JSONObject jsonObject = new JSONObject(inputLine);
                    JSONArray jsonArray = jsonObject.getJSONArray("images");

                    for (int i = 0; i < jsonArray.length(); i++) {

                        JSONArray jsonArray_fields = jsonArray.getJSONObject(i).getJSONArray("fields");

                        boolean questionFlag = true;
                        boolean answerFlag = false;
                        String q = "";
                        int m = 0, n = 0;

                        while (questionFlag) {
                            String inferText = jsonArray_fields.getJSONObject(m++).getString("inferText").replace(".", "");
                            if (isNumeric(inferText)) {
                                Log.d("aaaa", "aaaaaaaaaQuestion" + inferText);
                                q = inferText;
                                question.add(q);
                                questionFlag = false;
                            }
                        }

                        /*
                        과목이 수학이고 주관식인 경우에만 OCR로 답 확인이 가능함
                         */
                        if (subject.equals("수학") && Integer.parseInt(q) >= 21) {
                            while (!answerFlag) {
                                String inferText = jsonArray_fields.getJSONObject(n++).getString("inferText").replace(")", "").replace(":", "");
                                if (inferText.equals("답")) {
                                    Log.d("aaaa", "aaaaaaaaaMyAnswer" + jsonArray_fields.getJSONObject(n).getString("inferText"));
                                    myAnswer.put(q, jsonArray_fields.getJSONObject(n).getString("inferText"));
                                    answerFlag = true;
                                }
                            }
                        } else{
                            Log.d(TAG, "inside else");
                            File imgfile = new File(imageFile);
                            tempAnswer = inferenceImage(i, imgfile);
                            Log.d(TAG, "temp answer is "+tempAnswer);
                            myAnswer.put(q, tempAnswer);
                        }
                    }
                } catch (Exception e) {
                }

                br.close();

            } catch (Exception e) {
                System.out.println(e);
            }
        } // END: run()
    } // END: ncp()

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    private static void writeMultiPart(OutputStream out, String jsonMessage, File file, String boundary) throws
            IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("--").append(boundary).append("\r\n");
        sb.append("Content-Disposition:form-data; name=\"message\"\r\n\r\n");
        sb.append(jsonMessage);
        sb.append("\r\n");
        out.write(sb.toString().getBytes("UTF-8"));
        out.flush();
        if (file != null && file.isFile()) {
            out.write(("--" + boundary + "\r\n").getBytes("UTF-8"));
            StringBuilder fileString = new StringBuilder();
            fileString
                    .append("Content-Disposition:form-data; name=\"file\"; filename=");
            fileString.append("\"" + file.getName() + "\"\r\n");
            fileString.append("Content-Type: application/octet-stream\r\n\r\n");
            out.write(fileString.toString().getBytes("UTF-8"));
            out.flush();
            try (FileInputStream fis = new FileInputStream(file)) {
                byte[] buffer = new byte[8192];
                int count;
                while ((count = fis.read(buffer)) != -1) {
                    out.write(buffer, 0, count);
                }
                out.write("\r\n".getBytes());
            }
            out.write(("--" + boundary + "--\r\n").getBytes("UTF-8"));
        }
        out.flush();
    }

    // 문자열이 숫자로 되었는지 판단
    public boolean isNumeric(String input) {
        try {
            Double.parseDouble(input);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    public class GetAnswer extends Thread {

        private String question;

        public GetAnswer(String question) {
            this.question = question;
        }

        @Override
        public void run() {
            String serverUri = "http://ec2-13-125-208-213.ap-northeast-2.compute.amazonaws.com/getAnswer.php";
            String parameters = "grade=" + grade + "&year=" + year + "&month=" + month + "&subject=" + subject + "&question=" + question;
            try {
                URL url = new URL(serverUri);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setDoInput(true);
                connection.setUseCaches(false);
                OutputStream outputStream = connection.getOutputStream();
                outputStream.write(parameters.getBytes("UTF-8"));
                outputStream.flush();
                outputStream.close();
                InputStream is = connection.getInputStream();
                InputStreamReader isr = new InputStreamReader(is);
                BufferedReader reader = new BufferedReader(isr);
                final StringBuffer buffer = new StringBuffer();
                String line = reader.readLine();
                while (line != null) {
                    buffer.append(line + "\n");
                    line = reader.readLine();
                }
                //읽어온 문자열에서 row(레코드)별로 분리하여 배열로 리턴하기
                String tempResult = buffer.toString();
                String result = "";

                    /*
                    buffer.toString으로 받아온 걸 바로 쓰니까 결과값이 잘못 나와서
                    result란 변수를 새로 만들어서 isDigit인 것으로만 결과를 새로 만드니 잘 됨
                     */
                    for (char t : tempResult.toCharArray()) {
                    if (Character.isDigit(t)) {
                        result += t;
                    }
                }
                qanda.put(question, result);
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public class InsertWrong extends Thread {

        private String question;

        public InsertWrong(String question) {
            this.question = question;
        }

        @Override
        public void run() {
            String serverUri = "http://ec2-13-125-208-213.ap-northeast-2.compute.amazonaws.com/insertWrong.php";
            String parameters = "user="+ User.email+"&grade=" + grade + "&year=" + year + "&month=" + month + "&subject=" + subject + "&question=" + this.question;
            try {
                URL url = new URL(serverUri);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setDoInput(true);
                connection.setUseCaches(false);

                OutputStream outputStream = connection.getOutputStream();
                outputStream.write(parameters.getBytes("UTF-8"));
                outputStream.flush();
                outputStream.close();

                InputStream is=connection.getInputStream();
                InputStreamReader isr= new InputStreamReader(is);
                BufferedReader reader= new BufferedReader(isr);
                final StringBuffer buffer= new StringBuffer();
                String line= reader.readLine();
                while (line!=null){
                    buffer.append(line+"\n");
                    line= reader.readLine();
                }
                //읽어온 문자열에서 row(레코드)별로 분리하여 배열로 리턴하기
                String result = buffer.toString();
                Log.d("zzzzzzzzzzzz","zzzzzzzzzzResult"+ result);
            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public class GetMockId extends Thread {

        @Override
        public void run() {
            String serverUri = "http://ec2-13-125-208-213.ap-northeast-2.compute.amazonaws.com/getMockId.php";
            String parameters = "grade=" + grade + "&year=" + year + "&month=" + month + "&subject=" + subject;
            try {
                URL url = new URL(serverUri);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setDoInput(true);
                connection.setUseCaches(false);
                OutputStream outputStream = connection.getOutputStream();
                outputStream.write(parameters.getBytes("UTF-8"));
                outputStream.flush();
                outputStream.close();
                InputStream is = connection.getInputStream();
                InputStreamReader isr = new InputStreamReader(is);
                BufferedReader reader = new BufferedReader(isr);
                final StringBuffer buffer = new StringBuffer();
                String line = reader.readLine();
                while (line != null) {
                    buffer.append(line + "\n");
                    line = reader.readLine();
                }
                //읽어온 문자열에서 row(레코드)별로 분리하여 배열로 리턴하기
                mock_id = buffer.toString();
                Log.d("aaaaaaaaaaa","aaaaaaaaaa"+mock_id);

            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public String inferenceImage(int i, File file) throws IOException {

        RequestBody requestFile =
                RequestBody.create(MediaType.parse("multipart/form-data"), file);

        // MultipartBody.Part is used to send also the actual file name
        MultipartBody.Part body =
                MultipartBody.Part.createFormData("image", file.getName(), requestFile);

        Call<String> call = retroApi.inferenceImage("sds", body);

        String result = call.execute().body();

        return result;

    }


}