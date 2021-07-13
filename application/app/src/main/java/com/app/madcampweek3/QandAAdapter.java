package com.app.madcampweek3;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import com.bumptech.glide.Glide;
import com.squareup.picasso.Picasso;

import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class QandAAdapter extends RecyclerView.Adapter<QandAAdapter.QandAViewHolder> {
    private ArrayList<QandAItem> qandaList;

    public QandAAdapter(ArrayList<QandAItem> qaList){
        this.qandaList = qaList;
    }

    // 레이아웃 파일에 있는 UI 컴포넌트를 NoteViewHolder 클래스의 멤버 변수와 연결
    public class QandAViewHolder extends RecyclerView.ViewHolder{
        TextView qanda;
        ImageView q_image;
        ImageView explanation;

        public QandAViewHolder(View view){
            super(view);
            qanda = (TextView)view.findViewById(R.id.qanda);
            q_image = (ImageView)view.findViewById(R.id.imageQuestion);
            explanation = (ImageView)view.findViewById(R.id.imgExp);
        }
    }

    @NonNull
    @Override
    public QandAViewHolder onCreateViewHolder(@NonNull ViewGroup viewGroup, int viewType) {
        View view = LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.qanda_item,viewGroup,false);
        return new QandAAdapter.QandAViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull final QandAViewHolder viewHolder, int position) {
        final QandAItem qandAItem = qandaList.get(position);
        final boolean[] openFlag = {false};

        viewHolder.qanda.setText(qandAItem.getQanda());
        Picasso.get().load(qandAItem.getQ_image()).into(viewHolder.q_image);
        Picasso.get().load(qandAItem.getExplanation()).into(viewHolder.explanation);

        viewHolder.q_image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(!openFlag[0]){
                    viewHolder.explanation.setVisibility(View.VISIBLE);
                    openFlag[0] = true;
                }else{
                    viewHolder.explanation.setVisibility(View.GONE);
                    openFlag[0] = false;
                }
            }
        });
    }

    @Override
    public int getItemCount() {
        return (null != qandaList ? qandaList.size() : 0);
    }
}

